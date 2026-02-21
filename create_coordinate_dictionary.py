import pandas as pd
import re
from collections import defaultdict

# =========================
# 1) LOAD YOUR FILE
# =========================
PATH = "Symptoms_Data_SymMap_SMTS.xlsx"
SHEET = 0
ENCODING = None

if PATH.lower().endswith((".xlsx", ".xls")):
    df = pd.read_excel(PATH, sheet_name=SHEET)
else:
    df = pd.read_csv(PATH, encoding=ENCODING or "utf-8")

print("Columns:", list(df.columns))

# =========================
# 2) PICK THE RIGHT TEXT COLUMN
# =========================
# Prefer explicit pattern/syndrome columns if they exist.
CANDIDATES_PATTERN = [
    "pattern", "Pattern", "证候", "病机", "辨证", "证型", "syndrome", "Syndrome"
]
# Otherwise fall back to symptom text columns (your file has these).
CANDIDATES_SYMPTOM_TEXT = [
    "Symptom_definition", "symptom_definition",
    "TCM_symptom_name", "TCM_symptom", "Symptom_name", "symptom_name"
]

PATTERN_COL = None
for c in CANDIDATES_PATTERN:
    if c in df.columns:
        PATTERN_COL = c
        break

if PATTERN_COL is None:
    for c in CANDIDATES_SYMPTOM_TEXT:
        if c in df.columns:
            PATTERN_COL = c
            break

if PATTERN_COL is None:
    raise ValueError(
        "Couldn't find a usable text column. "
        "Please set PATTERN_COL manually to one of: " + ", ".join(map(str, df.columns))
    )

print("Using text column:", PATTERN_COL)

text_raw = df[PATTERN_COL].dropna().astype(str)

# =========================
# 2b) SPLITTING STRATEGY
# =========================
# If we're using an actual pattern/syndrome column, splitting on conjunctions might help.
# If we're using symptom definitions, splitting on 及/并/伴/兼 is TOO aggressive.
USING_PATTERN_COL = PATTERN_COL in CANDIDATES_PATTERN

if USING_PATTERN_COL:
    SPLIT_REGEX = r"[，,；;、/|]|及|并|伴|兼"
else:
    SPLIT_REGEX = r"[，,；;、/|]"  # punctuation only

def split_terms(text: str):
    parts = re.split(SPLIT_REGEX, text)
    return [p.strip() for p in parts if p.strip()]

all_terms = []
for cell in text_raw:
    all_terms.extend(split_terms(cell))

unique_terms = sorted(set(all_terms))
print("Unique terms extracted:", len(unique_terms))

# Save the extracted term list (so you can inspect if extraction makes sense)
pd.Series(unique_terms, name="term").to_csv(
    "unique_patterns_from_dataset.csv", index=False, encoding="utf-8-sig"
)
print("Saved: unique_patterns_from_dataset.csv")

# =========================
# 3) RULE LEXICONS
# =========================
HOT = ["热", "火", "炎", "燥", "毒", "温", "灼", "赤", "黄", "热盛", "火旺", "实热", "湿热", "痰热", "血热"]
COLD = ["寒", "冷", "凉", "清", "稀", "白", "寒盛", "虚寒", "内寒", "寒凝", "寒湿"]
DEF = ["虚", "不足", "亏", "衰", "弱", "不固", "失约", "下陷", "脱", "不振", "不荣", "不摄", "不纳"]
EXC = ["实", "盛", "亢", "壅", "郁", "滞", "逆", "闭", "阻", "积", "结", "上逆", "内闭"]

EXT = ["表", "外感", "风寒", "风热", "风湿", "暑", "燥邪", "疫", "外袭", "犯表", "束表", "卫分"]
INT = ["里", "内伤", "脏", "腑", "气滞", "血瘀", "痰", "湿", "食积", "水饮", "瘀", "结"]
ORGANS = ["肝", "心", "脾", "肺", "肾", "胃", "胆", "膀胱", "大肠", "小肠", "子宫", "胞宫", "冲任", "带脉", "任脉", "督脉"]

YIN_HINTS  = ["阴", "寒", "虚寒", "阳虚", "血虚", "津亏", "精亏", "不固", "失约"]
YANG_HINTS = ["阳", "热", "实热", "火", "亢", "旺", "热盛"]

def has_any(term, keywords):
    return any(k in term for k in keywords)

# =========================
# 4) LABEL EACH TERM
# =========================
labels = {k: set() for k in ["hot","cold","internal","external","deficiency","excess","yin","yang"]}
combo = defaultdict(set)

def assign_labels(term: str):
    term_labels = set()

    # hot/cold
    if has_any(term, HOT): term_labels.add("hot")
    if has_any(term, COLD): term_labels.add("cold")

    # deficiency/excess
    if has_any(term, DEF): term_labels.add("deficiency")
    if has_any(term, EXC): term_labels.add("excess")

    # external/internal
    if has_any(term, EXT): term_labels.add("external")
    if has_any(term, ORGANS) or has_any(term, INT) or ("里" in term) or ("内" in term):
        term_labels.add("internal")

    # yin/yang (heuristic)
    if has_any(term, YIN_HINTS) or ("cold" in term_labels) or ("deficiency" in term_labels):
        term_labels.add("yin")
    if has_any(term, YANG_HINTS) or ("hot" in term_labels) or ("excess" in term_labels):
        term_labels.add("yang")

    return term_labels

for t in unique_terms:
    labs = assign_labels(t)
    for lab in labs:
        labels[lab].add(t)
    combo[tuple(sorted(labs))].add(t)

EIGHT_PRINCIPLES_DICT = {k: sorted(v) for k, v in labels.items()}

# Useful combos only
COMBO_DICT = {}
for k, terms in combo.items():
    if not k:
        continue
    if (("hot" in k or "cold" in k) and ("deficiency" in k or "excess" in k)):
        COMBO_DICT["_".join(k)] = sorted(terms)

print("\n==== EIGHT PRINCIPLES COUNTS ====")
for k in ["hot","cold","internal","external","deficiency","excess","yin","yang"]:
    print(k, len(EIGHT_PRINCIPLES_DICT[k]))

print("\n==== COMBO COUNT ====")
print(len(COMBO_DICT))

# =========================
# 5) SAVE TO CSV (what you asked for)
# =========================
# Multi-label format: one row per (term,label)
eight_rows = []
for label, terms in EIGHT_PRINCIPLES_DICT.items():
    for term in terms:
        eight_rows.append({"term": term, "label": label})

pd.DataFrame(eight_rows).to_csv("eight_principles_terms.csv", index=False, encoding="utf-8-sig")
print("Saved: eight_principles_terms.csv")

combo_rows = []
for combo_label, terms in COMBO_DICT.items():
    for term in terms:
        combo_rows.append({"term": term, "combo_label": combo_label})

pd.DataFrame(combo_rows).to_csv("combination_terms.csv", index=False, encoding="utf-8-sig")
print("Saved: combination_terms.csv")