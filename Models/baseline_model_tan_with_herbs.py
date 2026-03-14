"""
Owner: Lianne

Baseline Model: TAN (Tree-Augmented Naive Bayes)

Builds and evaluates a TAN baseline for TCM syndrome prediction, using a syndrome-symptom matrix,
eight-principles mappings and symptom location mappings.

Output:
- Bar chart: overall metrics (accuracy, macro-F1, top-5)
- Learning curve: performance vs training set size
"""

import os
import re
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx


# Take a SMTS column name and extract the integer ID
# example: SMTS001 -> 1
def smts_to_int(colname):
    num = re.fullmatch(r"SMTS(\d+)", colname.strip())
    if not num:
        return None
    return int(num.group(1))


# Turn counts into probabilities with Laplace smoothing
def laplace_smooth_counts(counts, alpha=1.0):
    counts = counts.astype(float)
    return (counts + alpha) / (counts.sum() + alpha * len(counts))


# Avoid log(0) issues
def safe_log(x):
    x = np.clip(x, 1e-12, 1.0)
    return np.log(x)


# compute top-k accuracy given predicted probabilities and true labels
def topk_accuracy(probs, y_true, k=5):
    topk = np.argsort(-probs, axis=1)[:, :k]
    hits = []
    for i in range(len(y_true)):
        hits.append(y_true[i] in topk[i])
    return float(np.mean(hits))


# Confusion matrix: counts how many times each true class was predicted as each predicted class
def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for i in range(len(y_true)):
        yt = y_true[i]
        yp = y_pred[i]
        cm[yt, yp] += 1
    return cm


# Use confusion matrix to compute macro-F1 score (unweighted average of per-class F1 scores)
# F1 score: how well a model balances avoiding false positives (precision) and avoiding false negatives (recall)
def macro_f1_from_cm(cm):
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0).astype(float) - tp
    fn = cm.sum(axis=1).astype(float) - tp

    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp + fn, 1)
    denom = np.maximum(precision + recall, 1e-12)
    f1 = 2 * precision * recall / denom

    return float(f1.mean())


def load_data(syndrome_symptom_csv, eight_principles_csv, location_csv):
    ss = pd.read_csv(syndrome_symptom_csv)
    syndrome_ids = ss["Syndrome_id"].astype(str).tolist()

    # grab only the symptom columns (skip Syndrome_id)
    kept_cols = [c for c in ss.columns if smts_to_int(c) is not None]
    smts_ids = [smts_to_int(c) for c in kept_cols]

    X_syndrome_sym = ss[kept_cols].fillna(0).values.astype(np.int8)

    # load the two feature tables and align them to our symptom order
    ep = pd.read_csv(eight_principles_csv).set_index("TCM_symptom_id")
    loc = pd.read_csv(location_csv).set_index("TCM_symptom_id")

    ep_aligned = ep.reindex(smts_ids).fillna(0).astype(int)
    loc_aligned = loc.reindex(smts_ids).fillna(0).astype(int)

    return syndrome_ids, smts_ids, X_syndrome_sym, ep_aligned, loc_aligned, kept_cols


# takes patient-symptom matrix and aggregates up to patient-eight_principles and patient-location matrices, then concatenates them together
def build_aggregated_features(X_symptoms, ep_aligned, loc_aligned):
    ep_mat = ep_aligned.to_numpy(dtype=np.int8)
    loc_mat = loc_aligned.to_numpy(dtype=np.int8)

    ep_counts = (X_symptoms @ ep_mat).astype(np.int16)
    loc_counts = (X_symptoms @ loc_mat).astype(np.int16)

    # concatenate symptom features with the new aggregated features
    X_aug = np.hstack([X_symptoms, ep_counts, loc_counts]).astype(np.int16)
    ep_cols = [f"count_{c}" for c in ep_aligned.columns]
    loc_cols = [f"count_{c}" for c in loc_aligned.columns]

    return X_aug, ep_cols, loc_cols


# Compute mutual information between each binary feature and the class label
# Return indices of top features
def mutual_information(X_bin, y, max_features, alpha=1.0):
    N, D = X_bin.shape
    C = int(y.max()) + 1
    y_counts = np.bincount(y, minlength=C).astype(float)
    py = (y_counts + alpha) / (N + alpha * C)

    mi = np.zeros(D, dtype=float)

    for i in range(D):
        xi = X_bin[:, i]
        table = np.zeros((C, 2), dtype=float)
        for c in range(C):
            mask = (y == c)
            table[c, 0] = np.sum(xi[mask] == 0)
            table[c, 1] = np.sum(xi[mask] == 1)

        pxy = (table + alpha) / (N + alpha * C * 2)
        px = pxy.sum(axis=0)

        val = 0.0
        for c in range(C):
            for x in (0, 1):
                p = pxy[c, x]
                denom = px[x] * py[c]
                val += p * np.log(p / denom)
        mi[i] = val

    idx = np.argsort(-mi)[:max_features]
    return idx


# Compute conditional mutual information between two binary features given the class label
def conditional_mutual_information_binary(Xi, Xj, y, num_classes, alpha=1.0):
    cmi = 0.0
    N = len(y)
    for c in range(num_classes):
        mask = (y == c)
        Nc = int(mask.sum())
        if Nc == 0:
            continue

        xi = Xi[mask]
        xj = Xj[mask]

        joint = np.zeros((2, 2), dtype=float)
        for a in (0, 1):
            for b in (0, 1):
                joint[a, b] = np.sum((xi == a) & (xj == b))

        joint = (joint + alpha) / (Nc + 4 * alpha)
        pi = joint.sum(axis=1, keepdims=True)
        pj = joint.sum(axis=0, keepdims=True)

        for a in (0, 1):
            for b in (0, 1):
                p_ab = joint[a, b]
                denom = pi[a, 0] * pj[0, b]
                cmi += (Nc / N) * p_ab * np.log(p_ab / denom)
    return float(cmi)


class TANClassifier:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.num_classes = None
        self.class_prior = None
        self.parent = None
        self.root = 0
        self.p_root = None
        self.p_child = None

    def fit(self, X_bin, y):
        N, D = X_bin.shape
        y = y.astype(int)
        C = int(y.max()) + 1
        self.num_classes = C

        counts = np.bincount(y, minlength=C)
        self.class_prior = laplace_smooth_counts(counts, alpha=self.alpha)

        # build the complete graph and find the maximum spanning tree
        G = nx.Graph()
        G.add_nodes_from(range(D))

        # compute CMI for each pair of features and add as edge weights
        for i in range(D):
            Xi = X_bin[:, i]
            for j in range(i + 1, D):
                w = conditional_mutual_information_binary(Xi, X_bin[:, j], y, C, alpha=self.alpha)
                G.add_edge(i, j, weight=w)

        T = nx.maximum_spanning_tree(G, weight="weight")

        # BFS to assign parents
        parent = {self.root: -1}
        queue = [self.root]
        while queue:
            u = queue.pop(0)
            for v in T.neighbors(u):
                if v not in parent:
                    parent[v] = u
                    queue.append(v)
        self.parent = np.array([parent[i] for i in range(D)], dtype=int)

        # estimate CPTs with laplace smoothing
        self.p_root = np.zeros((C, 2), dtype=float)
        for c in range(C):
            mask = (y == c)
            Nc = int(mask.sum())
            for xval in (0, 1):
                cnt = np.sum(X_bin[mask, self.root] == xval)
                self.p_root[c, xval] = (cnt + self.alpha) / (Nc + 2 * self.alpha)

        self.p_child = np.zeros((C, D, 2, 2), dtype=float)
        for i in range(D):
            if i == self.root:
                continue
            p = self.parent[i]
            for c in range(C):
                mask = (y == c)
                for pv in (0, 1):
                    mask_pv = mask & (X_bin[:, p] == pv)
                    Ncpv = int(mask_pv.sum())
                    for xv in (0, 1):
                        cnt = np.sum(mask_pv & (X_bin[:, i] == xv))
                        self.p_child[c, i, pv, xv] = (cnt + self.alpha) / (Ncpv + 2 * self.alpha)

        return self

    def predict_proba(self, X_bin):
        N, D = X_bin.shape
        C = self.num_classes
        log_prior = safe_log(self.class_prior)

        logp = np.zeros((N, C), dtype=float)
        for c in range(C):
            lp = np.full(N, log_prior[c], dtype=float)
            lp += safe_log(self.p_root[c, X_bin[:, self.root]])

            for i in range(D):
                if i == self.root:
                    continue
                p = self.parent[i]
                lp += safe_log(self.p_child[c, i, X_bin[:, p], X_bin[:, i]])

            logp[:, c] = lp

        # softmax-style normalization
        logp -= logp.max(axis=1, keepdims=True)
        proba = np.exp(logp)
        proba /= proba.sum(axis=1, keepdims=True)
        return proba

    def predict(self, X_bin):
        return self.predict_proba(X_bin).argmax(axis=1)


def load_herb_data(syndrome_herb_file, syndrome_ids):
    # read the syndrome-herb matrix and make sure rows are in the same order as syndrome_ids
    herb_df = pd.read_csv(syndrome_herb_file, index_col=0)
    herb_df.index = herb_df.index.astype(str)
    syndrome_ids = [str(x) for x in syndrome_ids]
    herb_df = herb_df.reindex(syndrome_ids).fillna(0)
    herb_ids = herb_df.columns.tolist()
    syndrome_herb_matrix = herb_df.values.astype(np.float32)
    return syndrome_herb_matrix, herb_ids


def recommend_herbs_tan(syndrome_probs, syndrome_herb_matrix, top_k=5):
    # weigh each herb by how probable its syndromes are, then take the top k
    herb_scores = syndrome_probs @ syndrome_herb_matrix
    return np.argsort(-herb_scores)[:top_k]


def evaluate_herb_recommendations(proba, y_true, syndrome_herb_matrix, k_values=(1, 3, 5, 7, 10)):
    mean_precisions = []
    mean_recalls = []

    for k in k_values:
        precisions = []
        recalls = []

        for i in range(len(y_true)):
            top_indices = set(recommend_herbs_tan(proba[i], syndrome_herb_matrix, top_k=k).tolist())
            # ground truth: whichever herbs are listed for this patient's actual syndrome
            true_indices = set(np.where(syndrome_herb_matrix[y_true[i]] > 0)[0].tolist())

            true_pos = len(top_indices & true_indices)
            precisions.append(true_pos / len(top_indices) if top_indices else 0)
            recalls.append(true_pos / len(true_indices) if true_indices else 0)

        mean_precisions.append(float(np.mean(precisions)))
        mean_recalls.append(float(np.mean(recalls)))
        print(f"k={k:2d}  |  Mean Precision@k: {np.mean(precisions):.4f}  |  Mean Recall@k: {np.mean(recalls):.4f}")

    return mean_precisions, mean_recalls


def plot_herb_pk_curve(out_dir, k_values, mean_precisions, mean_recalls):
    # show how precision drops and recall rises as we recommend more herbs
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, mean_precisions, marker='o', linewidth=2, label='Precision@k')
    plt.plot(k_values, mean_recalls, marker='s', linewidth=2, label='Recall@k')

    # label each point so the values are easy to read
    for k, p, r in zip(k_values, mean_precisions, mean_recalls):
        plt.annotate(f'{p:.2f}', (k, p), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=9)
        plt.annotate(f'{r:.2f}', (k, r), textcoords='offset points', xytext=(0, -14), ha='center', fontsize=9)

    plt.title('Herb Recommendation Precision@k and Recall@k (TAN)', fontsize=15)
    plt.xlabel('k (number of herbs recommended)', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(k_values)
    plt.ylim(0, 1.1)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "herb_pk_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()


def save_predictions_csv(out_dir, proba, y_true, y_pred, syndrome_ids):
    conf = proba[np.arange(len(y_true)), y_pred]
    df = pd.DataFrame({
        "true_syndrome": [syndrome_ids[i] for i in y_true],
        "predicted_syndrome": [syndrome_ids[i] for i in y_pred],
        "confidence": conf,
    })
    df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)


def save_top_confusions_csv(out_dir, y_true, y_pred, syndrome_ids):
    pairs = Counter((int(yt), int(yp)) for yt, yp in zip(y_true, y_pred) if yt != yp)
    rows = [(syndrome_ids[yt], syndrome_ids[yp], cnt) for (yt, yp), cnt in pairs.items()]
    df = pd.DataFrame(rows, columns=["true", "predicted", "count"]).sort_values("count", ascending=False)
    df.to_csv(os.path.join(out_dir, "top_confusions.csv"), index=False)


def save_herb_predictions_csv(out_dir, proba, y_true, syndrome_herb_matrix, herb_ids, k=5):
    rows = []

    for i in range(len(y_true)):
        herb_scores = proba[i] @ syndrome_herb_matrix
        top_idx = np.argsort(-herb_scores)[:k]
        true_idx = np.where(syndrome_herb_matrix[y_true[i]] > 0)[0]

        rows.append({
            "patient_index": i,
            "true_syndrome_index": int(y_true[i]),
            "true_herbs": ",".join([str(herb_ids[j]) for j in true_idx]),
            "top_predicted_herbs": ",".join([str(herb_ids[j]) for j in top_idx]),
            "top_predicted_scores": ",".join([f"{herb_scores[j]:.4f}" for j in top_idx]),
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "herb_predictions_topk.csv"), index=False)


# bar chart of overall test metrics
def plot_overall_metrics_bar(out_dir, metrics, label="TAN"):
    names = ["Accuracy", "Top-5 Accuracy", "Macro-F1"]
    vals = [metrics["accuracy"], metrics["top5_accuracy"], metrics["macro_f1"]]

    plt.figure(figsize=(7.5, 4.8))
    x = np.arange(len(names))
    plt.bar(x, vals)
    plt.xticks(x, names)
    plt.ylim(0.0, 1.0)
    plt.title("Overall Test Metrics")

    for i, v in enumerate(vals):
        plt.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "overall_metrics.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # also save metrics as CSV for easy comparison with other models
    pd.DataFrame([{"model": label, **metrics}]).to_csv(os.path.join(out_dir, "metrics_tan.csv"), index=False)


def plot_learning_curve(out_dir, Xtr, ytr, Xte, yte, fractions=(0.1, 0.25, 0.5, 0.75, 1.0), seed=0):
    rng = np.random.default_rng(seed)
    N = len(ytr)
    order = np.arange(N)
    rng.shuffle(order)

    rows = []
    for frac in fractions:
        n = max(2, int(frac * N))
        idx = order[:n]

        model = TANClassifier(alpha=1.0).fit(Xtr[idx], ytr[idx])
        proba = model.predict_proba(Xte)
        yhat = proba.argmax(axis=1)

        C = int(max(yte.max(), yhat.max())) + 1
        cm = confusion_matrix(yte, yhat, C)
        rows.append({
            "train_frac": float(frac),
            "train_n": int(n),
            "accuracy": float((yhat == yte).mean()),
            "top5_accuracy": float(topk_accuracy(proba, yte, k=5)),
            "macro_f1": float(macro_f1_from_cm(cm)),
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "learning_curve.csv"), index=False)

    plt.figure(figsize=(7.6, 5.0))
    plt.plot(df["train_n"], df["accuracy"], marker="o", label="Accuracy")
    plt.plot(df["train_n"], df["macro_f1"], marker="o", label="Macro-F1")
    plt.plot(df["train_n"], df["top5_accuracy"], marker="o", label="Top-5 Acc")
    plt.title("Learning Curve (TAN)")
    plt.xlabel("Number of training examples")
    plt.ylabel("Metric")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "learning_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()


def load_patient_data(patient_symptoms_csv, patient_labels_csv, smts_ids):
    sym_df = pd.read_csv(patient_symptoms_csv, header=0)
    sym_df.columns = [int(c) for c in sym_df.columns]
    sym_df = sym_df.reindex(columns=smts_ids, fill_value=0)
    X_pat_sym = sym_df.values.astype(np.int8)

    lab_df = pd.read_csv(patient_labels_csv)
    y = lab_df["Syndrome_id"].values.astype(int)

    return X_pat_sym, y


def run_experiment(
    syndrome_symptom_csv,
    eight_principles_csv,
    location_csv,
    patient_symptoms_csv,
    patient_labels_csv,
    syndrome_herb_file=None,
    out_dir="outputs",
    seed=0,
    max_features=200,
):
    os.makedirs(out_dir, exist_ok=True)

    syndrome_ids, smts_ids, X_syndrome_sym, ep_aligned, loc_aligned, symptom_colnames = load_data(
        syndrome_symptom_csv, eight_principles_csv, location_csv
    )

    # load real patient data and augment features
    X_pat_sym, y = load_patient_data(patient_symptoms_csv, patient_labels_csv, smts_ids)
    X_aug, ep_cols, loc_cols = build_aggregated_features(X_pat_sym, ep_aligned, loc_aligned)
    feat_names = list(symptom_colnames) + list(ep_cols) + list(loc_cols)

    X_bin = (X_aug > 0).astype(np.int8)
    N, D = X_bin.shape

    # 80/20 train/test split
    rng = np.random.default_rng(seed)
    idx = np.arange(N)
    rng.shuffle(idx)
    split = int(0.8 * N)
    tr, te = idx[:split], idx[split:]
    Xtr, ytr = X_bin[tr], y[tr]
    Xte, yte = X_bin[te], y[te]

    # feature screening to keep things tractable
    if max_features is not None and D > max_features:
        feat_idx = mutual_information(Xtr, ytr, max_features=max_features)
        Xtr_s = Xtr[:, feat_idx]
        Xte_s = Xte[:, feat_idx]
        feat_names_sub = [feat_names[i] for i in feat_idx]
    else:
        feat_idx = np.arange(D)
        Xtr_s, Xte_s = Xtr, Xte
        feat_names_sub = feat_names

    # fit and evaluate
    model = TANClassifier(alpha=1.0).fit(Xtr_s, ytr)
    proba = model.predict_proba(Xte_s)
    yhat = proba.argmax(axis=1)

    acc = float((yhat == yte).mean())
    top5 = float(topk_accuracy(proba, yte, k=5))
    cm = confusion_matrix(yte, yhat, num_classes=int(y.max()) + 1)
    mf1 = float(macro_f1_from_cm(cm))

    print(f"Test accuracy:  {acc:.4f}")
    print(f"Test top-5 acc: {top5:.4f}")
    print(f"Test macro-F1:  {mf1:.4f}")

    metrics = {"accuracy": acc, "top5_accuracy": top5, "macro_f1": mf1}

    herb_results = None
    if syndrome_herb_file is not None:
        syndrome_herb_matrix, herb_ids = load_herb_data(syndrome_herb_file, syndrome_ids)

        print("\nHerb recommendation results:")
        k_values = (1, 3, 5, 7, 10)
        mean_precisions, mean_recalls = evaluate_herb_recommendations(
            proba, yte, syndrome_herb_matrix, k_values=k_values
        )

        plot_herb_pk_curve(out_dir, k_values, mean_precisions, mean_recalls)
        save_herb_predictions_csv(out_dir, proba, yte, syndrome_herb_matrix, herb_ids, k=5)

        herb_results = {
            "k_values": k_values,
            "precision_at_k": mean_precisions,
            "recall_at_k": mean_recalls,
            "herb_ids": herb_ids,
        }

        pd.DataFrame({
            "k": k_values,
            "precision_at_k": mean_precisions,
            "recall_at_k": mean_recalls,
        }).to_csv(os.path.join(out_dir, "herb_metrics.csv"), index=False)

    # save outputs
    save_predictions_csv(out_dir, proba, yte, yhat, syndrome_ids)
    save_top_confusions_csv(out_dir, yte, yhat, syndrome_ids)
    plot_overall_metrics_bar(out_dir, metrics, label="TAN")
    plot_learning_curve(out_dir, Xtr_s, ytr, Xte_s, yte, fractions=(0.1, 0.25, 0.5, 0.75, 1.0), seed=seed)

    return {
        "model": model,
        "proba": proba,
        "yte": yte,
        "yhat": yhat,
        "syndrome_ids": syndrome_ids,
        "feat_names_sub": feat_names_sub,
        "feat_idx": feat_idx,
        "metrics": metrics,
        "herb_results": herb_results,
    }


if __name__ == "__main__":
    run_experiment(
        syndrome_symptom_csv="Processed Datasets/Final_Training_Features_Syndrome_Symptom.csv",
        eight_principles_csv="Legacy/SMTS_eight_principles_by_id.csv",
        location_csv="Processed Datasets/Symptom_Location_Features.csv",
        patient_symptoms_csv="Patient Datasets/Synthetic_Patient_Symptoms.csv",
        patient_labels_csv="Patient Datasets/Synthetic_Patient_Labels.csv",
        syndrome_herb_file="Processed Datasets/Syndrome_Herb_Targets.csv",
        out_dir="outputs",
        seed=0,
        max_features=200,
    )