# concatenate_datasets.py
#
# Builds a final training table by combining:
#  1) Final_Training_Features_Syndrome_Symptom.csv  (wide: syndrome_id + symptom columns, 0/1)
#  2) Symptom_Location_Features.csv                  (wide: symptom_id + feature columns, 0/1)
#  3) terms_onehot.csv                               (wide: term + label_* columns)   [optional]
#  4) combination_terms_onehot.csv                   (wide: term + combo_* columns)   [optional]
#
# Output:
#  - Final_Training_Features_Syndrome_Symptom_With_Location.csv (wide: syndrome_id + symptom_feature columns)
#  - Final_Training_Edges_Long.csv (long edge table: syndrome_id, TCM_symptom_id, present, + symptom features)

import pandas as pd
from pathlib import Path


# ---------------------------
# Config (filenames)
# ---------------------------
EDGES_WIDE_PATH = Path("Final_Training_Features_Syndrome_Symptom.csv")
SYMPTOM_FEATURES_PATH = Path("Symptom_Location_Features.csv")

# Optional: these are NOT merged into the syndrome-symptom matrix (different entity),
# but we load + sanity-check so your pipeline is in one script.
TERMS_ONEHOT_PATH = Path("terms_onehot.csv")
COMBO_ONEHOT_PATH = Path("combination_terms_onehot.csv")

OUT_WIDE_PATH = Path("Final_Training_Features_Syndrome_Symptom_With_Location.csv")
OUT_LONG_PATH = Path("Final_Training_Edges_Long.csv")


# ---------------------------
# Helpers
# ---------------------------
def _pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column name in candidates that exists in df, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _ensure_str_id(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df[col] = df[col].astype(str)
    return df


def _coerce_01(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Coerce columns to int 0/1 safely."""
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)
    return out


def build_final_syndrome_feature_table(
    edges_wide_path: Path,
    symptom_features_path: Path,
    out_wide_path: Path,
    out_long_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # ---------------------------
    # 1) Load syndrome->symptom wide matrix
    # ---------------------------
    edges_wide = pd.read_csv(edges_wide_path, encoding="utf-8", engine="python")

    # syndrome id column could be "Syndrome_id" or "syndrome_id" etc.
    syndrome_id_col = _pick_first_existing(
        edges_wide,
        ["Syndrome_id", "syndrome_id", "syndrome", "SyndromeID", "Syndrome Id", "Syndrome"],
    )
    if syndrome_id_col is None:
        raise ValueError(
            f"Could not find syndrome id column in {edges_wide_path.name}. "
            f"Columns found: {edges_wide.columns.tolist()[:30]}..."
        )

    edges_wide = edges_wide.rename(columns={syndrome_id_col: "Syndrome_id"})
    edges_wide = _ensure_str_id(edges_wide, "Syndrome_id")

    # Symptom columns are everything except Syndrome_id
    symptom_cols = [c for c in edges_wide.columns if c != "Syndrome_id"]
    if len(symptom_cols) == 0:
        raise ValueError(f"No symptom columns found in {edges_wide_path.name} after identifying Syndrome_id.")

    # Coerce symptom presence to 0/1
    edges_wide = _coerce_01(edges_wide, symptom_cols)

    # ---------------------------
    # 2) Load symptom feature table
    # ---------------------------
    symptom_features = pd.read_csv(symptom_features_path, encoding="utf-8", engine="python")

    # Identify symptom id col. In your screenshot, it's likely "TCM_symptom_id",
    # but we handle common variants.
    symptom_id_col = _pick_first_existing(
        symptom_features,
        ["TCM_symptom_id", "tcm_symptom_id", "symptom_id", "Symptom_id", "SymptomID", "TCM Symptom Id"],
    )
    if symptom_id_col is None:
        raise ValueError(
            f"Could not find symptom id column in {symptom_features_path.name}. "
            f"Columns found: {symptom_features.columns.tolist()}"
        )

    # Standardize key to TCM_symptom_id
    if symptom_id_col != "TCM_symptom_id":
        symptom_features = symptom_features.rename(columns={symptom_id_col: "TCM_symptom_id"})
    symptom_features = _ensure_str_id(symptom_features, "TCM_symptom_id")

    # Feature columns: everything except key
    feat_cols = [c for c in symptom_features.columns if c != "TCM_symptom_id"]
    if len(feat_cols) == 0:
        raise ValueError(f"No feature columns found in {symptom_features_path.name} besides the id column.")

    symptom_features = _coerce_01(symptom_features, feat_cols)

    # ---------------------------
    # 3) Convert wide edges -> long edges (syndrome_id, symptom_id, present)
    # ---------------------------
    # Melt: one row per (syndrome, symptom) with presence value
    A_long = edges_wide.melt(
        id_vars=["Syndrome_id"],
        value_vars=symptom_cols,
        var_name="TCM_symptom_id",
        value_name="present",
    )

    # Coerce ID types (important for merge)
    A_long = _ensure_str_id(A_long, "TCM_symptom_id")
    A_long["present"] = pd.to_numeric(A_long["present"], errors="coerce").fillna(0).astype(int)

    # Optionally keep only present edges (often desired)
    A_long_present = A_long[A_long["present"] == 1].copy()

    # ---------------------------
    # 4) Attach symptom features onto the long edges
    # ---------------------------
    # IMPORTANT: this will fail if symptom_features lacks TCM_symptom_id — we standardize above.
    full_long = A_long_present.merge(symptom_features, on="TCM_symptom_id", how="left")

    # Fill missing features with 0 (if a symptom id wasn't found in feature table)
    for c in feat_cols:
        if c in full_long.columns:
            full_long[c] = full_long[c].fillna(0).astype(int)

    # Save long table
    full_long.to_csv(out_long_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_long_path}")

    # ---------------------------
    # 5) Build a wide syndrome feature table:
    #     for each syndrome, sum feature counts across present symptoms
    # ---------------------------
    # Group by syndrome and sum each feature column (counts of present symptoms with that feature)
    wide_features = full_long.groupby("Syndrome_id")[feat_cols].sum().reset_index()

    # Optionally: keep original symptom presence columns too (so you still have the raw matrix)
    final_wide = edges_wide.merge(wide_features, on="Syndrome_id", how="left")

    # Fill NaNs if syndrome had no present symptoms (shouldn't happen, but safe)
    for c in feat_cols:
        final_wide[c] = final_wide[c].fillna(0).astype(int)

    # Save final wide table
    final_wide.to_csv(out_wide_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_wide_path}")

    return final_wide, full_long


def load_optional_onehots():
    # These are not merged into the syndrome/symptom table (different entity: terms),
    # but we load them so you can confirm they exist & look right.
    if TERMS_ONEHOT_PATH.exists():
        terms = pd.read_csv(TERMS_ONEHOT_PATH, encoding="utf-8", engine="python")
        print(f"Loaded: {TERMS_ONEHOT_PATH}  shape={terms.shape}")
    else:
        print(f"Optional missing (ok): {TERMS_ONEHOT_PATH}")

    if COMBO_ONEHOT_PATH.exists():
        combos = pd.read_csv(COMBO_ONEHOT_PATH, encoding="utf-8", engine="python")
        print(f"Loaded: {COMBO_ONEHOT_PATH}  shape={combos.shape}")
    else:
        print(f"Optional missing (ok): {COMBO_ONEHOT_PATH}")


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    # Optional sanity-check loads
    load_optional_onehots()

    # Build concatenated datasets for syndrome/symptom + symptom features
    final_wide, full_long = build_final_syndrome_feature_table(
        edges_wide_path=EDGES_WIDE_PATH,
        symptom_features_path=SYMPTOM_FEATURES_PATH,
        out_wide_path=OUT_WIDE_PATH,
        out_long_path=OUT_LONG_PATH,
    )

    # Quick debug prints
    print("\n--- DONE ---")
    print("Final wide shape:", final_wide.shape)
    print("Final long (present edges) shape:", full_long.shape)