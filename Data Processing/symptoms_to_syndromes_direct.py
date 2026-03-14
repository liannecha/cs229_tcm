"""
Owner: Lianne
**Based off of cs229TCMLocation.py**

This code extracts eight-principles features from the SMTS dataset.

It creates binary features indicating whether a symptom is associated with the TCM
eight principles:

- hot
- cold
- internal
- external
- yin
- yang
- excess
- deficiency

Keyword matching is performed over the symptom name, pinyin, definition, and locus.

Output:
Symptom_Eight_Principles_Features.csv
"""

import pandas as pd
import numpy as np

# Load SMTS dataset
smts_df = pd.read_excel("Symptoms_Data_SymMap_SMTS.xlsx")
smts_df = smts_df.fillna("")

# Combine relevant text fields into one searchable column
smts_df["Eight_Principles_Text"] = (
    smts_df["TCM_symptom_name"].astype(str) + " " +
    smts_df["Symptom_pinYin"].astype(str) + " " +
    smts_df["Symptom_definition"].astype(str) + " " +
    smts_df["Symptom_locus"].astype(str)
).str.lower()

# Dictionary of keywords related to the eight principles
eight_principles_map = {

    "hot": [
        "热", "发热", "高热", "低热", "潮热", "壮热", "身热",
        "烦热", "烦躁", "面红", "目赤", "口渴",
        "fever", "heat", "hot", "burning", "thirst"
    ],

    "cold": [
        "寒", "怕冷", "恶寒", "畏寒", "寒战", "肢冷", "四肢冷",
        "冷痛", "喜暖", "得温则减",
        "cold", "chills", "cold limbs", "prefers warmth"
    ],

    "internal": [
        "里", "内", "脏", "腑", "胸", "腹", "胃", "肠",
        "心", "肝", "脾", "肺", "肾",
        "internal", "interior", "organ", "viscera"
    ],

    "external": [
        "表", "外", "肌表", "皮毛", "皮肤",
        "恶风", "恶寒", "头痛", "身痛", "鼻塞",
        "external", "exterior", "surface"
    ],

    "yin": [
        "阴", "虚寒", "内寒", "面色淡", "疲惫",
        "肢冷", "喜静", "尿清", "便溏",
        "yin", "cold", "deficiency cold"
    ],

    "yang": [
        "阳", "热", "实热", "面红", "烦躁",
        "口渴", "脉数", "声高",
        "yang", "heat", "excess heat"
    ],

    "excess": [
        "实", "实证", "胀满", "拒按", "疼痛剧烈",
        "痰多", "便秘", "尿闭", "高热",
        "excess", "fullness", "distention"
    ],

    "deficiency": [
        "虚", "虚证", "乏力", "疲倦",
        "气短", "自汗", "盗汗",
        "面色淡", "隐痛", "喜按",
        "deficiency", "weak", "fatigue"
    ]
}

for principle, keywords in eight_principles_map.items():
    smts_df[principle] = smts_df["Eight_Principles_Text"].apply(
        lambda x: 1 if any(keyword in x for keyword in keywords) else 0
    )

output_df = smts_df[
    ["TCM_symptom_id", "hot", "cold", "internal", "external", "yin", "yang", "excess", "deficiency"]
]

output_df.to_csv("Symptom_Eight_Principles_Features.csv", index=False)

print("Created Symptom_Eight_Principles_Features.csv")