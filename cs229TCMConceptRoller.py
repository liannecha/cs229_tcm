import pandas as pd

# Load primary feature matrix (233 Syndromes x 1717 Symptoms)
X_symptoms = pd.read_csv("Final_Training_Features_Syndrome_Symptom.csv", index_col=0)

# Force symptom column headers to be integers so they match index of the other datasets
X_symptoms.columns = X_symptoms.columns.astype(int)

# Load the other datasets
df_locations = pd.read_csv("Symptom_Location_Features.csv", index_col='TCM_symptom_id')
df_coords = pd.read_csv("SMTS_eight_principles_by_id.csv", index_col='TCM_symptom_id')

# Combine into one dataframe
df_combined_kangae = pd.concat([df_locations, df_coords], axis=1)

# Align index to be safe
df_combined_kangae = df_combined_kangae.reindex(X_symptoms.columns).fillna(0)

# Now matrix multiplication to unroll the features into a single matrix
syndrome_kangae = X_symptoms.dot(df_combined_kangae)

# One hot
syndrome_kangae = (syndrome_kangae > 0).astype(int)

# save
syndrome_kangae.to_csv("Syndrome_Concept_Targets.csv")
print("Did it!")
print(syndrome_kangae.head())
