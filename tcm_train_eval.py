"""
tcm_train_eval.py
=================
Complete training, evaluation, ablation, and hyperparameter-search pipeline
for the CS229 TCM herb prediction project.

Usage (runs everything end-to-end):
    python tcm_train_eval.py

What this script does:
  1. Loads data and builds the herb co-occurrence heatmap.
  2. Trains four model variants (ablation study):
       a. TANBaseline   — linear, no hidden layers
       b. PlainMLP      — MLP, no concept bottleneck
       c. ConceptBM     — Concept Bottleneck Model
       d. ConceptBM_CNN — CBM + CNN herb compatibility module  ← main model
  3. For ConceptBM_CNN also reports:
       • baseline (no CNN) vs CNN results side-by-side
  4. Evaluates herb prediction:
       • Precision@k and Recall@k for k ∈ {3, 5, 10}
       • Compatibility score (% predicted pairs that appear in training data)
  5. Runs a small hyperparameter grid search and records best config.
  6. Plots and saves:
       • herb_cooccurrence_heatmap.png
       • loss_curves_<model>.png  for each variant
       • confusion_matrix.png     for concept prediction
       • ablation_results.png     bar chart comparing all variants
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# --- Project modules ---
from tcm_dataset import (
    build_training_herb_pairs,
    compatibility_score,
    compute_herb_cooccurrence,
    load_tcm_data,
    make_train_val_split,
    normalize_cooccurrence_row,
    plot_cooccurrence_heatmap,
    precision_recall_at_k,
    TCMDataset,
)
from tcm_model import (
    ConceptBM,
    ConceptBM_CNN,
    PlainMLP,
    TANBaseline,
    tcm_loss,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = Path(".")   # all plots/results land in working directory

DEFAULT_HP = dict(lr=1e-3, dropout=0.3, hidden_dim=256)
EPOCHS = 150
BATCH_SIZE = 32  # full dataset is only 228 rows; mini-batches still help regularise
SEED = 42

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Plotting (non-interactive backend)
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def _make_model(
    name: str,
    input_dim: int,
    n_concepts: int,
    n_herbs: int,
    hp: dict,
) -> nn.Module:
    hd = hp["hidden_dim"]
    dr = hp["dropout"]
    if name == "TANBaseline":
        return TANBaseline(input_dim, n_concepts, n_herbs)
    if name == "PlainMLP":
        return PlainMLP(input_dim, n_concepts, n_herbs, hidden_dim=hd, dropout=dr)
    if name == "ConceptBM":
        return ConceptBM(input_dim, n_concepts, n_herbs, hidden_dim=hd, dropout=dr)
    if name == "ConceptBM_CNN":
        return ConceptBM_CNN(input_dim, n_concepts, n_herbs, hidden_dim=hd, dropout=dr)
    raise ValueError(f"Unknown model name: {name}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    use_cnn: bool = True,
) -> dict:
    model.train()
    totals: Dict[str, float] = defaultdict(float)
    n_batches = 0

    for x, y_con, y_herb in loader:
        x, y_con, y_herb = x.to(DEVICE), y_con.to(DEVICE), y_herb.to(DEVICE)
        optimizer.zero_grad()

        if isinstance(model, ConceptBM_CNN):
            con_logits, herb_logits, base_logits = model(x, use_cnn=use_cnn)
        else:
            con_logits, herb_logits, base_logits = model(x)

        loss, comps = tcm_loss(con_logits, herb_logits, y_con, y_herb,
                               baseline_herb_logits=base_logits)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        for k, v in comps.items():
            totals[k] += v
        totals["total"] += loss.item()
        n_batches += 1

    return {k: v / n_batches for k, v in totals.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    use_cnn: bool = True,
) -> dict:
    model.eval()
    totals: Dict[str, float] = defaultdict(float)
    n_batches = 0

    all_herb_logits: List[torch.Tensor] = []
    all_herb_targets: List[torch.Tensor] = []
    all_con_logits: List[torch.Tensor] = []
    all_con_targets: List[torch.Tensor] = []

    for x, y_con, y_herb in loader:
        x, y_con, y_herb = x.to(DEVICE), y_con.to(DEVICE), y_herb.to(DEVICE)

        if isinstance(model, ConceptBM_CNN):
            con_logits, herb_logits, base_logits = model(x, use_cnn=use_cnn)
        else:
            con_logits, herb_logits, base_logits = model(x)

        loss, comps = tcm_loss(con_logits, herb_logits, y_con, y_herb,
                               baseline_herb_logits=base_logits)
        for k, v in comps.items():
            totals[k] += v
        totals["total"] += loss.item()
        n_batches += 1

        all_herb_logits.append(herb_logits.cpu())
        all_herb_targets.append(y_herb.cpu())
        all_con_logits.append(con_logits.cpu())
        all_con_targets.append(y_con.cpu())

    herb_probs = torch.sigmoid(torch.cat(all_herb_logits)).numpy()
    herb_targets = torch.cat(all_herb_targets).numpy()
    con_probs = torch.sigmoid(torch.cat(all_con_logits)).numpy()
    con_targets = torch.cat(all_con_targets).numpy()

    results = {k: v / n_batches for k, v in totals.items()}
    results["herb_probs"] = herb_probs
    results["herb_targets"] = herb_targets
    results["con_probs"] = con_probs
    results["con_targets"] = con_targets
    return results


def compute_concept_accuracy(con_probs: np.ndarray, con_targets: np.ndarray) -> float:
    preds = (con_probs >= 0.5).astype(float)
    return float((preds == con_targets).mean())


def compute_herb_metrics(
    herb_probs: np.ndarray,
    herb_targets: np.ndarray,
    training_pairs: set,
) -> dict:
    metrics = {}
    for k in [3, 5, 10]:
        p, r = precision_recall_at_k(herb_probs, herb_targets, k)
        metrics[f"P@{k}"] = round(p, 4)
        metrics[f"R@{k}"] = round(r, 4)
    metrics["compat@5"] = round(compatibility_score(herb_probs, training_pairs, k=5), 4)
    metrics["compat@10"] = round(compatibility_score(herb_probs, training_pairs, k=10), 4)
    return metrics


# ---------------------------------------------------------------------------
# Full training run for one model
# ---------------------------------------------------------------------------

def run_training(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_dim: int,
    n_concepts: int,
    n_herbs: int,
    hp: dict,
    epochs: int = EPOCHS,
    label: str = "",
) -> dict:
    """Train a model; return metrics and loss history."""
    model = _make_model(model_name, input_dim, n_concepts, n_herbs, hp).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hp["lr"], weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses, val_losses = [], []
    train_concept_losses, val_concept_losses = [], []
    train_herb_losses, val_herb_losses = [], []

    best_val_loss = float("inf")
    best_state: dict | None = None

    for epoch in range(1, epochs + 1):
        t_comps = train_one_epoch(model, train_loader, optimizer)
        v_comps = evaluate(model, val_loader)
        scheduler.step()

        train_losses.append(t_comps["total"])
        val_losses.append(v_comps["total"])
        train_concept_losses.append(t_comps.get("concept", 0.0))
        val_concept_losses.append(v_comps.get("concept", 0.0))
        train_herb_losses.append(t_comps.get("herb", 0.0))
        val_herb_losses.append(v_comps.get("herb", 0.0))

        if v_comps["total"] < best_val_loss:
            best_val_loss = v_comps["total"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 25 == 0:
            print(
                f"  [{label or model_name}] Epoch {epoch:3d} | "
                f"train={t_comps['total']:.4f}  val={v_comps['total']:.4f}"
            )

    # Restore best weights
    if best_state:
        model.load_state_dict(best_state)

    return dict(
        model=model,
        train_losses=train_losses,
        val_losses=val_losses,
        train_concept_losses=train_concept_losses,
        val_concept_losses=val_concept_losses,
        train_herb_losses=train_herb_losses,
        val_herb_losses=val_herb_losses,
        best_val_loss=best_val_loss,
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_loss_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_concept: List[float],
    val_concept: List[float],
    train_herb: List[float],
    val_herb: List[float],
    title: str,
    save_path: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(1, len(train_losses) + 1)

    for ax, (tr, vl, name) in zip(
        axes,
        [
            (train_losses, val_losses, "Total Loss"),
            (train_concept, val_concept, "Concept Loss"),
            (train_herb, val_herb, "Herb Loss"),
        ],
    ):
        ax.plot(epochs, tr, label="Train", linewidth=1.5)
        ax.plot(epochs, vl, label="Val", linewidth=1.5, linestyle="--")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BCE Loss")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"[plot] Saved → {save_path}")


def plot_ablation_results(results_table: dict, save_path: str) -> None:
    models = list(results_table.keys())
    metrics = ["P@3", "P@5", "P@10", "R@3", "R@5", "R@10", "compat@5", "compat@10"]

    x = np.arange(len(models))
    n_metrics = len(metrics)
    width = 0.10
    offsets = np.linspace(-(n_metrics // 2) * width, (n_metrics // 2) * width, n_metrics)

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (metric, offset) in enumerate(zip(metrics, offsets)):
        vals = [results_table[m].get(metric, 0) for m in models]
        ax.bar(x + offset, vals, width=width, label=metric, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("Score")
    ax.set_title("Ablation Study — Herb Prediction Metrics")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"[plot] Saved → {save_path}")


def plot_confusion_matrix(
    con_probs: np.ndarray,
    con_targets: np.ndarray,
    concept_cols: List[str],
    save_path: str,
) -> None:
    from sklearn.metrics import multilabel_confusion_matrix
    preds = (con_probs >= 0.5).astype(int)
    # Per-concept: TP/FP/FN/TN
    n = len(concept_cols)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(14, 6))
    axes = axes.flatten()

    for i, (col, ax) in enumerate(zip(concept_cols, axes)):
        tp = int(((preds[:, i] == 1) & (con_targets[:, i] == 1)).sum())
        fp = int(((preds[:, i] == 1) & (con_targets[:, i] == 0)).sum())
        fn = int(((preds[:, i] == 0) & (con_targets[:, i] == 1)).sum())
        tn = int(((preds[:, i] == 0) & (con_targets[:, i] == 0)).sum())
        mat = np.array([[tn, fp], [fn, tp]])
        ax.imshow(mat, cmap="Blues", vmin=0)
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(mat[r, c]), ha="center", va="center", fontsize=10)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred 0", "Pred 1"], fontsize=7)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["True 0", "True 1"], fontsize=7)
        ax.set_title(col, fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Per-concept Confusion Matrices (val set)", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"[plot] Saved → {save_path}")


def plot_hyperparam_results(hp_results: List[dict], save_path: str) -> None:
    labels = [
        f"lr={r['lr']}\ndo={r['dropout']}\nhd={r['hidden_dim']}"
        for r in hp_results
    ]
    val_losses = [r["best_val_loss"] for r in hp_results]
    p5_scores = [r["P@5"] for r in hp_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(labels))

    ax1.bar(x, val_losses, color="steelblue", alpha=0.8)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=7)
    ax1.set_title("Val Total Loss by Hyperparams")
    ax1.set_ylabel("Loss")
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x, p5_scores, color="darkorange", alpha=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=7)
    ax2.set_title("Val Precision@5 by Hyperparams")
    ax2.set_ylabel("P@5")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Hyperparameter Grid Search — ConceptBM_CNN", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"[plot] Saved → {save_path}")


# ---------------------------------------------------------------------------
# Error analysis helper
# ---------------------------------------------------------------------------

def syndrome_error_analysis(
    con_probs: np.ndarray,
    con_targets: np.ndarray,
    syndrome_ids: List[str],
    concept_cols: List[str],
    top_n: int = 10,
) -> str:
    """
    Identify the top-N most confused syndrome samples (highest mean
    concept-label hamming error per sample) and return a summary string.
    """
    preds = (con_probs >= 0.5).astype(float)
    per_sample_errors = np.abs(preds - con_targets).mean(axis=1)
    top_idx = np.argsort(per_sample_errors)[-top_n:][::-1]

    lines = ["=== Error Analysis: Top confused syndromes (val set) ==="]
    for rank, idx in enumerate(top_idx, 1):
        sid = syndrome_ids[idx] if idx < len(syndrome_ids) else f"idx={idx}"
        wrong_concepts = [
            concept_cols[j]
            for j in range(len(concept_cols))
            if preds[idx, j] != con_targets[idx, j]
        ]
        lines.append(
            f"  {rank:2d}. {sid}  hamming_err={per_sample_errors[idx]:.3f}"
            f"  wrong_labels={wrong_concepts}"
        )
    return "\n".join(lines)


def herb_error_analysis(
    herb_probs: np.ndarray,
    herb_targets: np.ndarray,
    syndrome_ids: List[str],
    herb_ids: List[str],
    k: int = 5,
    top_n: int = 5,
) -> str:
    """
    Find syndromes where top-k predicted herbs overlap least with ground truth.
    """
    lines = ["=== Error Analysis: Worst herb predictions (val set) ==="]
    scores = []
    for i in range(len(herb_probs)):
        top_k = set(np.argsort(herb_probs[i])[-k:])
        true_set = set(np.where(herb_targets[i] > 0.5)[0])
        overlap = len(top_k & true_set)
        scores.append(overlap / k)

    worst_idx = np.argsort(scores)[:top_n]
    for rank, idx in enumerate(worst_idx, 1):
        sid = syndrome_ids[idx] if idx < len(syndrome_ids) else f"idx={idx}"
        top_k = list(np.argsort(herb_probs[idx])[-k:])
        true_herbs = list(np.where(herb_targets[idx] > 0.5)[0])
        pred_names = [herb_ids[i] if i < len(herb_ids) else str(i) for i in top_k]
        true_names = [herb_ids[i] if i < len(herb_ids) else str(i) for i in true_herbs[:5]]
        lines.append(
            f"  {rank}. {sid}  P@{k}={scores[idx]:.2f}"
            f"\n     Pred: {pred_names}"
            f"\n     True: {true_names}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Hyperparameter grid search
# ---------------------------------------------------------------------------

def run_hyperparam_search(
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_dataset_indices: List[int],
    full_dataset: TCMDataset,
    training_pairs: set,
    input_dim: int,
    n_concepts: int,
    n_herbs: int,
    epochs: int = 80,
) -> List[dict]:
    """
    Small grid over lr × dropout × hidden_dim.
    Returns list of result dicts sorted by val P@5.
    """
    grid = list(product(
        [1e-3, 1e-4],        # lr
        [0.2, 0.5],          # dropout
        [128, 256, 512],     # hidden_dim
    ))
    print(f"\n[hyperparam] Grid size: {len(grid)} configs on ConceptBM_CNN")

    results = []
    for lr, dropout, hidden_dim in grid:
        hp = dict(lr=lr, dropout=dropout, hidden_dim=hidden_dim)
        label = f"lr={lr} dr={dropout} hd={hidden_dim}"
        print(f"  Trying {label}")
        run = run_training(
            "ConceptBM_CNN", train_loader, val_loader,
            input_dim, n_concepts, n_herbs, hp,
            epochs=epochs, label=label,
        )
        val_out = evaluate(run["model"], val_loader)
        metrics = compute_herb_metrics(
            val_out["herb_probs"], val_out["herb_targets"], training_pairs
        )
        results.append(dict(
            lr=lr, dropout=dropout, hidden_dim=hidden_dim,
            best_val_loss=run["best_val_loss"],
            **metrics,
        ))

    results.sort(key=lambda r: -r["P@5"])
    return results


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("CS229 TCM — Herb Output Layer Training & Evaluation")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n[1/6] Loading data ...")
    data = load_tcm_data()
    syndrome_ids = data["syndrome_ids"]
    X = data["X"]
    Y_concept = data["Y_concept"]
    Y_herb = data["Y_herb"]
    herb_ids = data["herb_ids"]
    concept_cols = data["concept_cols"]

    N, D_feat = X.shape
    n_concepts = Y_concept.shape[1]
    n_herbs = Y_herb.shape[1]
    print(f"  N={N}  D_feat={D_feat}  n_concepts={n_concepts}  n_herbs={n_herbs}")

    # ------------------------------------------------------------------
    # 2. Herb co-occurrence matrix + heatmap
    # ------------------------------------------------------------------
    print("\n[2/6] Computing herb co-occurrence matrix ...")
    cooc = compute_herb_cooccurrence(Y_herb)
    cooc_norm = normalize_cooccurrence_row(cooc)
    print(f"  Co-occurrence matrix shape: {cooc.shape}")
    print(f"  Mean conditional co-occurrence (top-40 herbs): "
          f"{cooc_norm[:40, :40].mean():.4f}")
    plot_cooccurrence_heatmap(
        cooc_norm, herb_ids, top_n=40,
        save_path=str(OUT_DIR / "herb_cooccurrence_heatmap.png"),
    )

    # ------------------------------------------------------------------
    # 3. Build datasets and loaders
    # ------------------------------------------------------------------
    print("\n[3/6] Splitting data ...")
    dataset = TCMDataset(X, Y_concept, Y_herb)
    train_set, val_set = make_train_val_split(dataset, val_frac=0.20, seed=SEED)
    print(f"  Train: {len(train_set)}  Val: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    # Build training herb pair set for compatibility scoring
    train_herb_targets = Y_herb[train_set.indices]
    training_pairs = build_training_herb_pairs(train_herb_targets)
    print(f"  Unique herb pairs in training: {len(training_pairs):,}")

    # Ground-truth concept labels for val set (for error analysis)
    val_syndrome_ids = [syndrome_ids[i] for i in val_set.indices]

    # ------------------------------------------------------------------
    # 4. Ablation study — train all 4 model variants
    # ------------------------------------------------------------------
    print("\n[4/6] Ablation study ...")

    ablation_configs = [
        ("TANBaseline",  DEFAULT_HP),
        ("PlainMLP",     DEFAULT_HP),
        ("ConceptBM",    DEFAULT_HP),
        ("ConceptBM_CNN", DEFAULT_HP),
    ]

    ablation_results: dict = {}
    all_runs: dict = {}

    for model_name, hp in ablation_configs:
        print(f"\n  --- {model_name} ---")
        run = run_training(
            model_name, train_loader, val_loader,
            D_feat, n_concepts, n_herbs, hp,
            epochs=EPOCHS, label=model_name,
        )
        all_runs[model_name] = run

        # Val metrics
        val_out = evaluate(run["model"], val_loader)
        herb_metrics = compute_herb_metrics(
            val_out["herb_probs"], val_out["herb_targets"], training_pairs
        )
        concept_acc = compute_concept_accuracy(
            val_out["con_probs"], val_out["con_targets"]
        )
        herb_metrics["concept_acc"] = round(concept_acc, 4)
        herb_metrics["val_loss"] = round(run["best_val_loss"], 4)
        ablation_results[model_name] = herb_metrics

        print(f"    Concept acc : {concept_acc:.4f}")
        print(f"    P@3={herb_metrics['P@3']}  P@5={herb_metrics['P@5']}  P@10={herb_metrics['P@10']}")
        print(f"    R@3={herb_metrics['R@3']}  R@5={herb_metrics['R@5']}  R@10={herb_metrics['R@10']}")
        print(f"    Compat@5={herb_metrics['compat@5']}  Compat@10={herb_metrics['compat@10']}")

        # Save loss curves
        plot_loss_curves(
            run["train_losses"], run["val_losses"],
            run["train_concept_losses"], run["val_concept_losses"],
            run["train_herb_losses"], run["val_herb_losses"],
            title=f"Loss Curves — {model_name}",
            save_path=str(OUT_DIR / f"loss_curves_{model_name}.png"),
        )

    # Also evaluate ConceptBM_CNN without the CNN module (baseline head only)
    print("\n  --- ConceptBM_CNN (baseline head, no CNN) ---")
    cnn_model = all_runs["ConceptBM_CNN"]["model"]
    val_out_base = evaluate(cnn_model, val_loader, use_cnn=False)
    base_metrics = compute_herb_metrics(
        val_out_base["herb_probs"], val_out_base["herb_targets"], training_pairs
    )
    base_metrics["concept_acc"] = round(
        compute_concept_accuracy(val_out_base["con_probs"], val_out_base["con_targets"]), 4
    )
    ablation_results["ConceptBM_Baseline"] = base_metrics
    print(f"    P@5={base_metrics['P@5']}  compat@5={base_metrics['compat@5']}")

    print("\n  --- ConceptBM_CNN (with CNN) vs baseline comparison ---")
    print(f"    {'Metric':<15} {'Baseline':>10} {'CNN':>10} {'Delta':>10}")
    cnn_metrics = ablation_results["ConceptBM_CNN"]
    for m in ["P@3", "P@5", "P@10", "R@5", "compat@5"]:
        b = base_metrics[m]
        c = cnn_metrics[m]
        print(f"    {m:<15} {b:>10.4f} {c:>10.4f} {c-b:>+10.4f}")

    # ------------------------------------------------------------------
    # 5. Hyperparameter search
    # ------------------------------------------------------------------
    print("\n[5/6] Hyperparameter grid search ...")
    hp_results = run_hyperparam_search(
        train_loader, val_loader, val_set.indices,
        dataset, training_pairs,
        D_feat, n_concepts, n_herbs,
        epochs=80,
    )
    best_hp = hp_results[0]
    print(f"\n  Best HP: lr={best_hp['lr']}  dropout={best_hp['dropout']}"
          f"  hidden_dim={best_hp['hidden_dim']}"
          f"  → P@5={best_hp['P@5']:.4f}  val_loss={best_hp['best_val_loss']:.4f}")
    plot_hyperparam_results(
        hp_results,
        save_path=str(OUT_DIR / "hyperparam_search.png"),
    )

    # ------------------------------------------------------------------
    # 6. Error analysis and confusion matrix (on best CBM-CNN model)
    # ------------------------------------------------------------------
    print("\n[6/6] Error analysis ...")
    val_out_final = evaluate(cnn_model, val_loader, use_cnn=True)

    print(syndrome_error_analysis(
        val_out_final["con_probs"], val_out_final["con_targets"],
        val_syndrome_ids, concept_cols,
    ))
    print()
    print(herb_error_analysis(
        val_out_final["herb_probs"], val_out_final["herb_targets"],
        val_syndrome_ids, herb_ids, k=5,
    ))

    plot_confusion_matrix(
        val_out_final["con_probs"], val_out_final["con_targets"],
        concept_cols,
        save_path=str(OUT_DIR / "confusion_matrix.png"),
    )

    # Ablation bar chart
    plot_ablation_results(
        ablation_results,
        save_path=str(OUT_DIR / "ablation_results.png"),
    )

    # ------------------------------------------------------------------
    # 7. Save summary JSON
    # ------------------------------------------------------------------
    summary = dict(
        ablation=ablation_results,
        best_hyperparams=dict(
            lr=best_hp["lr"],
            dropout=best_hp["dropout"],
            hidden_dim=best_hp["hidden_dim"],
            val_P_at_5=best_hp["P@5"],
            val_loss=best_hp["best_val_loss"],
        ),
        dataset=dict(N=N, D_feat=D_feat, n_concepts=n_concepts, n_herbs=n_herbs),
    )
    summary_path = OUT_DIR / "tcm_results_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] Results written to {summary_path}")

    print("\n" + "=" * 60)
    print("Summary of ablation results:")
    print(f"  {'Model':<22} {'P@5':>6} {'R@5':>6} {'Compat@5':>10} {'ConceptAcc':>12}")
    for mname, mres in ablation_results.items():
        print(
            f"  {mname:<22} {mres.get('P@5', 0):>6.4f} {mres.get('R@5', 0):>6.4f} "
            f"{mres.get('compat@5', 0):>10.4f} {mres.get('concept_acc', 0):>12.4f}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
