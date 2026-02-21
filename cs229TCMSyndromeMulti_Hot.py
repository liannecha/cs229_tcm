import pandas as pd

edges_df = pd.read_csv("Symptom_Syndrome_Edges.csv")

# Pivot table to multi-hot encode the symptoms for each syndrome
# Rows = Syndromes
# Columns = Symptoms
multi_hot_df = pd.crosstab(edges_df['Syndrome_id'], edges_df['TCM_symptom_id'])

multi_hot_df = (multi_hot_df > 0).astype(int)  # Convert counts to binary (0/1) for one-hot encoding

multi_hot_df.to_csv("Final_Training_Features_Syndrome_Symptom.csv")

print(f"Created multi-hot encoded features for {multi_hot_df.shape[0]} syndromes and {multi_hot_df.shape[1]} symptoms.")