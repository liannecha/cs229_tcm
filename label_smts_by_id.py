import pandas as pd

# ======================================
# FILE PATHS
# ======================================

DICT_PATH = "eight_principles_terms.csv"
DATA_PATH = "Symptoms_Data_SymMap_SMTS.xlsx"
OUT_PATH  = "SMTS_eight_principles_by_id.csv"


# ======================================
# 1) LOAD DICTIONARY
# ======================================

dict_df = pd.read_csv(DICT_PATH, encoding="utf-8-sig")

if "label" not in dict_df.columns or "term" not in dict_df.columns:
    raise ValueError("eight_principles_terms.csv must contain: label, term")

label_terms = dict_df.groupby("label")["term"].apply(list).to_dict()

eight_labels = [
    "hot",
    "cold",
    "internal",
    "external",
    "deficiency",
    "excess",
    "yin",
    "yang",
]


# ======================================
# 2) LOAD SMTS DATASET
# ======================================

df = pd.read_excel(DATA_PATH)

required_cols = ["TCM_symptom_id", "TCM_symptom_name", "Symptom_definition"]

for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing column: {c}")


# ======================================
# 3) COMBINE TEXT FOR MATCHING
# ======================================

df["full_text"] = (
    df["TCM_symptom_name"].fillna("").astype(str)
    + " "
    + df["Symptom_definition"].fillna("").astype(str)
)


# ======================================
# 4) LABEL FUNCTION
# ======================================

def label_text(text):
    result = {}

    for lab in eight_labels:
        terms = label_terms.get(lab, [])
        result[lab] = int(any(t in text for t in terms))

    return pd.Series(result)


# ======================================
# 5) APPLY LABELS
# ======================================

labels_df = df["full_text"].apply(label_text)

result = pd.concat(
    [df["TCM_symptom_id"], labels_df],
    axis=1
)


# ======================================
# 6) SAVE OUTPUT
# ======================================

result.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print("Saved →", OUT_PATH)