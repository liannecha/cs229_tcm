"""
Owner: Lianne

Creates a multi-hot encoded feature set for herbs based on their associated syndromes.

Input features:  syndrome symptoms
Output targets: herbs
"""
import pandas as pd

edges_df = pd.read_csv("Processed Datasets/Syndrome_Herb_Edges.csv")

# Pivot table to multi-hot encode the herbs for each syndrome
# Rows = Syndromes
# Columns = Herbs
multi_hot_df = pd.crosstab(edges_df['Syndrome_id'], edges_df['Herb_id'])
multi_hot_df = (multi_hot_df > 0).astype(int)  # Convert counts to binary (0/1) for one-hot encoding

syndrome_order = pd.read_csv("Processed Datasets/Final_Training_Features_Syndrome_Symptom.csv", index_col=0).index
multi_hot_df = multi_hot_df.reindex(syndrome_order, fill_value=0)

multi_hot_df.to_csv("Syndrome_Herb_Targets.csv")

print(f"Created multi-hot encoded targets for {multi_hot_df.shape[0]} syndromes and {multi_hot_df.shape[1]} herbs.")