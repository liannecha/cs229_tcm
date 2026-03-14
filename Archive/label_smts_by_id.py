"""
Owner: Lianne

Label the symptoms in the SMTS dataset with the eight principles 
(hot, cold, internal, external, deficiency, excess, yin, yang) 
based on a dictionary of terms associated with each principle. 

Output: CSV file where each TCM symptom ID is paired with eight binary diagnostic indicators.
"""
import pandas as pd

dict_df = pd.read_csv("eight_principles_terms.csv")
# convert CSV to dictionary mapping each label to a list of terms
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

# Load SMTS dataset
df = pd.read_excel("Symptoms_Data_SymMap_SMTS.xlsx")

# Combine symptom name and definition into one text field for labeling
# Avoid NaN issues by filling with empty string
df["full_text"] = (
    df["TCM_symptom_name"].fillna("") + " " + df["Symptom_definition"].fillna("")
)

"""
Input: string of symptom text
Output: eight principles feature vector (binary labels)
"""
def label_text(text):
    result = {}
    for label in eight_labels:
        terms = label_terms.get(label, [])
        # If any term appears in the text, assign 1, otherwise assign 0.
        result[label] = int(any(t in text for t in terms))
    return pd.Series(result)

# Apply label_text to each row of the df
label_rows = []
for text in df["full_text"]:
    labels = label_text(text)
    label_rows.append(labels)

labels_df = pd.DataFrame(label_rows)

# Combine the TCM_symptom_id with the labels into one df
result = pd.concat([df["TCM_symptom_id"], labels_df], axis=1)

# encoding="utf-8-sig" necessary; file contains Chinese characters and this encoding ensures they are saved correctly
result.to_csv("SMTS_eight_principles_by_id.csv", index=False, encoding="utf-8-sig")