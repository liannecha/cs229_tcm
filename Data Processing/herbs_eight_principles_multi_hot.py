"""
Owner: Lianne

This code creates multi-hot encoded eight-principles labels for herbs
from the SMHB dataset.

Each herb is labeled with binary features for:
- hot
- cold
- internal
- external
- yin
- yang
- excess
- deficiency

Output:
Herb_Eight_Principles_Multihot.csv
"""

import pandas as pd

# Load the SMHB herb dataset
herb_df = pd.read_excel("Original Datasets/SymMap v2.0, SMHB file.xlsx")
herb_df = herb_df.fillna("")

# Combine all text fields
herb_df["Eight_Principles_Text"] = (
    herb_df["Chinese_name"].astype(str) + " " +
    herb_df["Pinyin_name"].astype(str) + " " +
    herb_df["Latin_name"].astype(str) + " " +
    herb_df["English_name"].astype(str) + " " +
    herb_df["Properties_Chinese"].astype(str) + " " +
    herb_df["Properties_English"].astype(str) + " " +
    herb_df["Meridians_Chinese"].astype(str) + " " +
    herb_df["Meridians_English"].astype(str) + " " +
    herb_df["Class_Chinese"].astype(str) + " " +
    herb_df["Class_English"].astype(str) + " " +
    herb_df["Alias"].astype(str)
).str.lower()

# Dictionary for eight principles
eight_principles_map = {
    "hot": [
        "热", "温", "微温", "hot", "warm", "slightly warm",
        "温里", "温经", "补阳", "助阳", "回阳", "散寒",
        "warming interior", "channel-warming", "yang reinforcing"
    ],

    "cold": [
        "寒", "凉", "微寒", "微凉", "cold", "cool", "slightly cold",
        "清热", "清热解毒", "清虚热", "凉血", "泻火", "解毒",
        "heat-clearing", "antipyretic", "blood-cooling", "detoxicate"
    ],

    "internal": [
        "里", "内", "interior", "internal",
        "温里药", "补气药", "补血药", "补阴药", "补阳药", "理气药",
        "化湿药", "化痰药", "攻下药", "安神药", "收涩药",
        "warming interior drugs", "qi reinforcing drugs", "blood-tonifying medicinal",
        "yin-tonifying medicinal", "yang reinforcing drugs", "qi regulating drugs",
        "dampness removing drugs", "phlegresolving medicine",
        "offensive purgative medicinal", "tranquilizing medicinal"
    ],

    "external": [
        "表", "外", "肌表", "exterior", "external",
        "解表", "辛温解表", "辛凉解表",
        "pungent-warm exterior-releasing medicinal",
        "pungent cool diaphoretics", "diaphoretics"
    ],

    "yin": [
        "阴", "yin",
        "补阴", "养阴", "滋阴", "清虚热",
        "yin-tonifying", "nourish yin", "asthenic heat",
        "寒", "凉", "cold", "cool"
    ],

    "yang": [
        "阳", "yang",
        "补阳", "助阳", "回阳", "温里", "温经", "散寒",
        "yang reinforcing", "restore yang", "warming interior",
        "热", "温", "hot", "warm"
    ],

    "excess": [
        "实", "excess",
        "攻下", "泻下", "清热解毒", "活血祛瘀", "祛风湿", "燥湿",
        "涌吐", "解毒杀虫", "驱虫", "消食", "化痰", "开窍",
        "offensive purgative", "antipyretic detoxicate",
        "blood activating stasis removing", "wind-dampnessdispelling",
        "dampnessdrying", "emetic", "antiparasitic",
        "phlegresolving", "resuscitative stimulant"
    ],

    "deficiency": [
        "虚", "deficiency",
        "补气", "补血", "补阴", "补阳", "养阴", "益气",
        "qi reinforcing", "blood-tonifying", "yin-tonifying",
        "yang reinforcing", "tonifying", "reinforcing"
    ]
}

# Create binary multi-hot columns
for principle, keywords in eight_principles_map.items():
    herb_df[principle] = herb_df["Eight_Principles_Text"].apply(
        lambda x: 1 if any(keyword in x for keyword in keywords) else 0
    )

output_df = herb_df[
    [
        "HERBDB_ID",
        "hot",
        "cold",
        "internal",
        "external",
        "yin",
        "yang",
        "excess",
        "deficiency"
    ]
]

# Split rows with multiple herbs into separate rows
output_df["HERBDB_ID"] = output_df["HERBDB_ID"].astype(str).str.split("|")
output_df = output_df.explode("HERBDB_ID")
output_df = output_df[output_df["HERBDB_ID"] != ""]
output_df = output_df.reset_index(drop=True)

output_df.to_csv("Herb_Eight_Principles_Multihot.csv", index=False)
print("Created Herb_Eight_Principles_Multihot.csv")