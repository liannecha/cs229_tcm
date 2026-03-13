"""
Owner: Ethan

This code extracts the location of symptoms from the SMTS dataset and maps them to the TCM "zang-fu" system of organs.
It creates binary features indicating whether a symptom is associated with each organ group 
(Wood, Fire, Earth, Metal, Water, Reproductive) based on keyword matching in the symptom's locus, name, and definition. 
The resulting features are saved to Symptom_Location_Features.csv.
"""
import pandas as pd
import numpy as np

# We want to extract the locus / location of the symptom
# For example, "headache" would be "head", "stomach ache" would be "stomach", etc.
# We map this to the validated "zang-fu" system, which includes organs like "heart", "liver", "spleen", "lung", and "kidney".
# there are two different types of symptoms, yang and yin, which are associated with different organs
# however, the organs across yang and yin share the same element, so we group them for simplicity and possible pattern recognition

smts_df = pd.read_excel("SMTS_file.xlsx")
smts_df = smts_df.fillna('')
smts_df['Locus_Text'] = (smts_df['Symptom_locus'] + ' ' + smts_df['TCM_symptom_name'] + ' ' + smts_df['Symptom_definition']).str.lower()

# Define a mapping of symptoms to organs
location_map = {
    'Is_Wood': ['肝', '胆', 'liver', 'gallbladder', 'wood'],
    'Is_Fire': ['心', '小肠', 'heart', 'small intestine', 'fire'],
    'Is_Earth': ['脾', '胃', 'spleen', 'stomach', 'earth'],
    'Is_Metal': ['肺', '大肠', 'lung', 'large intestine', 'metal'],
    'Is_Water': ['肾', '膀胱', 'kidney', 'bladder', 'water'],
    'Is_Reproductive': ['子宫', '乳房', 'uterus', 'breast']
}

for col, keywords in location_map.items():
    smts_df[col] = smts_df['Locus_Text'].apply(lambda x: 1 if any(keyword in x for keyword in keywords) else 0)

# save the updated dataframe to a new CSV file
loc_features = smts_df[['TCM_symptom_id', 'Is_Wood', 'Is_Fire', 'Is_Earth', 'Is_Metal', 'Is_Water', 'Is_Reproductive']]
loc_features.to_csv("Symptom_Location_Features.csv", index=False)

print("Did it!")