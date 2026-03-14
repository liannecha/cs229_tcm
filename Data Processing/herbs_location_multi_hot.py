"""
Owner: Lianne

Maps each herb to TCM organ-element groups.

Creates binary features indicating whether a herb is associated with:
- Wood
- Fire
- Earth
- Metal
- Water
- Reproductive
"""

import pandas as pd

herb_df = pd.read_excel("Original Datasets/SymMap v2.0, SMHB file.xlsx")
herb_df = herb_df.fillna("")

# split multiple herb ids into separate rows
herb_df["HERBDB_ID"] = herb_df["HERBDB_ID"].astype(str).str.split("|")
herb_df = herb_df.explode("HERBDB_ID").reset_index(drop=True)
herb_df["HERBDB_ID"] = herb_df["HERBDB_ID"].astype(str).str.strip()

# combine useful text fields into one lowercase text field
herb_df["Herb_Text"] = (
    herb_df["Meridians_Chinese"].astype(str) + " " +
    herb_df["Meridians_English"].astype(str) + " " +
    herb_df["Class_Chinese"].astype(str) + " " +
    herb_df["Class_English"].astype(str) + " " +
    herb_df["Chinese_name"].astype(str) + " " +
    herb_df["English_name"].astype(str)
).str.lower()

# define a mapping of keywords to organ groups
location_map = {
    "Is_Wood": ["肝", "胆", "liver", "gallbladder", "wood"],
    "Is_Fire": ["心", "小肠", "心包", "三焦", "heart", "small intestine", "pericardium", "triple burner", "san jiao", "fire"],
    "Is_Earth": ["脾", "胃", "spleen", "stomach", "earth"],
    "Is_Metal": ["肺", "大肠", "lung", "large intestine", "metal"],
    "Is_Water": ["肾", "膀胱", "kidney", "bladder", "water"],
    "Is_Reproductive": ["子宫", "乳房", "uterus", "breast", "reproductive"]
}

# create binary features for each organ group based on keyword matching in the Herb_Text field
for col, keywords in location_map.items():
    herb_df[col] = herb_df["Herb_Text"].apply(
        lambda x: 1 if any(keyword in x for keyword in keywords) else 0
    )

output_df = herb_df[
    [
        "Herb_id",
        "Is_Wood",
        "Is_Fire",
        "Is_Earth",
        "Is_Metal",
        "Is_Water",
        "Is_Reproductive",
    ]
].copy()

output_df.to_csv("Herb_Location_Features.csv", index=False, encoding="utf-8-sig")