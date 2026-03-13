"""
tcm_dataset.py
==============
Data loading, preprocessing, and herb co-occurrence matrix computation
for the CS229 TCM project.

Data layout (all files in the same directory as this script):
  Final_Training_Features_Syndrome_Symptom_With_Location.csv
      228 syndromes × 1875 binary/count feature columns
  Syndrome_Concept_Targets.csv
      228 syndromes × 14 binary concept labels
  Syndrome_Herb_Edges.csv
      2110 (Syndrome_id, Herb_id) edge rows  ->  596 unique herbs
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

# ---------------------------------------------------------------------------
# Paths (relative to this file's directory)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent

FEATURES_PATH = _HERE / "Final_Training_Features_Syndrome_Symptom_With_Location.csv"
CONCEPTS_PATH = _HERE / "Syndrome_Concept_Targets.csv"
HERB_EDGES_PATH = _HERE / "Syndrome_Herb_Edges.csv"

CONCEPT_COLS = [
    "Is_Wood", "Is_Fire", "Is_Earth", "Is_Metal", "Is_Water", "Is_Reproductive",
    "hot", "cold", "internal", "external", "deficiency", "excess", "yin", "yang",
]


# ---------------------------------------------------------------------------
# CSV helpers (no pandas dependency at runtime)
# ---------------------------------------------------------------------------

def _read_csv_as_matrix(path: Path) -> Tuple[List[str], List[str], np.ndarray]:
    """Return (row_ids, col_names, float32_matrix) for an ID-first CSV."""
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    # First column is the row ID
    row_ids = [r[0] for r in rows]
    col_names = header[1:]
    data = np.array([[float(v) for v in r[1:]] for r in rows], dtype=np.float32)
    return row_ids, col_names, data


def _read_herb_edges(path: Path) -> Dict[str, List[str]]:
    """Return {syndrome_id: [herb_id, ...]} from the edge CSV."""
    mapping: Dict[str, List[str]] = {}
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["Syndrome_id"].strip()
            hid = row["Herb_id"].strip()
            mapping.setdefault(sid, []).append(hid)
    return mapping


# ---------------------------------------------------------------------------
# Core data-loading function
# ---------------------------------------------------------------------------

def load_tcm_data(
    features_path: Path = FEATURES_PATH,
    concepts_path: Path = CONCEPTS_PATH,
    herb_edges_path: Path = HERB_EDGES_PATH,
) -> dict:
    """
    Load and align all three data sources.

    Returns a dict with keys:
        syndrome_ids  : List[str]  length N
        X             : np.ndarray (N, D_feat)   float32
        Y_concept     : np.ndarray (N, 14)        float32
        Y_herb        : np.ndarray (N, H)         float32
        herb_ids      : List[str]  length H (sorted by frequency desc)
        herb_freq     : np.ndarray (H,) float32   (# syndromes each herb appears in)
        feat_names    : List[str]  D_feat feature names
        concept_cols  : List[str]  14 concept column names
    """
    # --- Features ---
    feat_ids, feat_names, X = _read_csv_as_matrix(features_path)

    # --- Concepts ---
    con_ids, con_cols, Y_concept_raw = _read_csv_as_matrix(concepts_path)

    # Reorder concept columns to canonical order
    col_idx = {c: i for i, c in enumerate(con_cols)}
    concept_order = [col_idx[c] for c in CONCEPT_COLS if c in col_idx]
    # Fall back to whatever order exists if columns differ
    if len(concept_order) == len(CONCEPT_COLS):
        Y_concept_raw = Y_concept_raw[:, concept_order]
    concept_col_names = [CONCEPT_COLS[i] for i, _ in enumerate(concept_order)] \
        if len(concept_order) == len(CONCEPT_COLS) else con_cols

    # Align rows: use feat_ids as master
    con_id_to_row = {sid: i for i, sid in enumerate(con_ids)}
    concept_aligned = np.zeros((len(feat_ids), Y_concept_raw.shape[1]), dtype=np.float32)
    for i, sid in enumerate(feat_ids):
        if sid in con_id_to_row:
            concept_aligned[i] = Y_concept_raw[con_id_to_row[sid]]

    # --- Herbs ---
    herb_edges = _read_herb_edges(herb_edges_path)

    # Build sorted herb list (by frequency, descending) for consistent ordering
    herb_counter: Dict[str, int] = {}
    for herbs in herb_edges.values():
        for h in herbs:
            herb_counter[h] = herb_counter.get(h, 0) + 1
    herb_ids = sorted(herb_counter.keys(), key=lambda h: -herb_counter[h])
    herb_freq = np.array([herb_counter[h] for h in herb_ids], dtype=np.float32)

    herb_idx = {h: i for i, h in enumerate(herb_ids)}
    H = len(herb_ids)
    N = len(feat_ids)

    Y_herb = np.zeros((N, H), dtype=np.float32)
    for i, sid in enumerate(feat_ids):
        for h in herb_edges.get(sid, []):
            if h in herb_idx:
                Y_herb[i, herb_idx[h]] = 1.0

    return dict(
        syndrome_ids=feat_ids,
        X=X,
        Y_concept=concept_aligned,
        Y_herb=Y_herb,
        herb_ids=herb_ids,
        herb_freq=herb_freq,
        feat_names=feat_names,
        concept_cols=concept_col_names if isinstance(concept_col_names, list) else list(concept_col_names),
    )


# ---------------------------------------------------------------------------
# Herb co-occurrence matrix
# ---------------------------------------------------------------------------

def compute_herb_cooccurrence(Y_herb: np.ndarray) -> np.ndarray:
    """
    Compute and return the H×H raw co-occurrence count matrix.

    Entry (i, j) = number of syndromes where herb i AND herb j both appear.
    """
    # Y_herb: (N, H) binary  ->  co-occ = Y_herb.T @ Y_herb  (H, H)
    return (Y_herb.T @ Y_herb).astype(np.float32)


def normalize_cooccurrence_row(cooc: np.ndarray) -> np.ndarray:
    """
    Row-normalize cooc matrix: entry (i, j) = P(herb j | herb i present).
    """
    row_sums = cooc.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)  # avoid div by zero
    return cooc / row_sums


def plot_cooccurrence_heatmap(
    cooc_norm: np.ndarray,
    herb_ids: List[str],
    top_n: int = 40,
    save_path: str = "herb_cooccurrence_heatmap.png",
) -> None:
    """
    Plot the top-N (most frequent) herbs co-occurrence heatmap and save to file.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns
        _have_sns = True
    except ImportError:
        _have_sns = False

    n = min(top_n, cooc_norm.shape[0])
    sub = cooc_norm[:n, :n]
    labels = [h.replace("SMHB", "") for h in herb_ids[:n]]

    fig, ax = plt.subplots(figsize=(14, 12))
    if _have_sns:
        sns.heatmap(
            sub, ax=ax, xticklabels=labels, yticklabels=labels,
            cmap="YlOrRd", vmin=0, vmax=1,
            linewidths=0.1, linecolor="grey",
        )
    else:
        im = ax.imshow(sub, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=7)
        plt.colorbar(im, ax=ax, label="P(col herb | row herb)")

    ax.set_title(
        f"Herb Co-occurrence (row-normalized) — top {n} most frequent herbs\n"
        "Entry (i,j) = P(herb j | herb i appears in prescription)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[heatmap] Saved → {save_path}")


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class TCMDataset(Dataset):
    """
    Each sample = (x, y_concept, y_herb) for one syndrome.
    """

    def __init__(self, X: np.ndarray, Y_concept: np.ndarray, Y_herb: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.Y_concept = torch.from_numpy(Y_concept).float()
        self.Y_herb = torch.from_numpy(Y_herb).float()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y_concept[idx], self.Y_herb[idx]


def make_train_val_split(
    dataset: TCMDataset,
    val_frac: float = 0.20,
    seed: int = 42,
) -> Tuple[Subset, Subset]:
    """Reproducible random 80/20 split."""
    rng = np.random.default_rng(seed)
    n = len(dataset)
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    val_idx = idx[:n_val].tolist()
    train_idx = idx[n_val:].tolist()
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


# ---------------------------------------------------------------------------
# Compatibility score helper
# ---------------------------------------------------------------------------

def build_training_herb_pairs(Y_herb_train: np.ndarray) -> set:
    """
    Return a set of (i, j) frozensets representing herb pairs that co-occur
    in at least one training syndrome.  i < j always.
    """
    pairs: set = set()
    for row in Y_herb_train:
        present = np.where(row > 0.5)[0]
        for a in range(len(present)):
            for b in range(a + 1, len(present)):
                pairs.add((int(present[a]), int(present[b])))
    return pairs


def compatibility_score(
    pred_herb_probs: np.ndarray,
    training_pairs: set,
    k: int = 5,
) -> float:
    """
    For every sample in pred_herb_probs (N, H), take top-k herbs,
    enumerate all C(k,2) pairs, and compute the fraction that appear
    in training_pairs.
    """
    n_total = 0
    n_compat = 0
    for row in pred_herb_probs:
        top_k = np.argsort(row)[-k:]
        for a in range(len(top_k)):
            for b in range(a + 1, len(top_k)):
                i, j = int(min(top_k[a], top_k[b])), int(max(top_k[a], top_k[b]))
                n_total += 1
                if (i, j) in training_pairs:
                    n_compat += 1
    return n_compat / n_total if n_total > 0 else 0.0


def precision_recall_at_k(
    pred_probs: np.ndarray,
    targets: np.ndarray,
    k: int,
) -> Tuple[float, float]:
    """
    Macro-averaged Precision@k and Recall@k over all samples.
    pred_probs : (N, H) float
    targets    : (N, H) binary float
    """
    precisions, recalls = [], []
    for i in range(len(pred_probs)):
        top_k_idx = np.argsort(pred_probs[i])[-k:]
        true_positives = targets[i, top_k_idx].sum()
        total_positive = targets[i].sum()
        precisions.append(true_positives / k)
        recalls.append(true_positives / total_positive if total_positive > 0 else 0.0)
    return float(np.mean(precisions)), float(np.mean(recalls))


if __name__ == "__main__":
    data = load_tcm_data()
    print("Loaded TCM data:")
    print(f"  N syndromes  : {len(data['syndrome_ids'])}")
    print(f"  D features   : {data['X'].shape[1]}")
    print(f"  # concepts   : {data['Y_concept'].shape[1]}")
    print(f"  # herbs      : {data['Y_herb'].shape[1]}")
    print(f"  Herb IDs[0:5]: {data['herb_ids'][:5]}")

    cooc = compute_herb_cooccurrence(data["Y_herb"])
    cooc_norm = normalize_cooccurrence_row(cooc)
    print(f"  Co-occ matrix: {cooc.shape}")

    plot_cooccurrence_heatmap(cooc_norm, data["herb_ids"], top_n=40)
