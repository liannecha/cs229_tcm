import pandas as pd

# ==========================================================
# ONE-HOT ENCODE: EIGHT PRINCIPLES TERMS
# ==========================================================

terms_df = pd.read_csv("eight_principles_terms.csv", encoding="utf-8", engine="python")

terms_onehot = pd.get_dummies(terms_df["label"], prefix="label")

terms_encoded = pd.concat([terms_df[["term"]], terms_onehot], axis=1)

terms_encoded.to_csv("terms_onehot.csv", index=False, encoding="utf-8-sig")

print("Saved: terms_onehot.csv")


# ==========================================================
# ONE-HOT ENCODE: COMBINATION TERMS
# ==========================================================

combo_df = pd.read_csv("combination_terms.csv", encoding="utf-8", engine="python")

combo_onehot = pd.get_dummies(combo_df["combo_label"], prefix="combo")

combo_encoded = pd.concat([combo_df[["term"]], combo_onehot], axis=1)

combo_encoded.to_csv("combination_terms_onehot.csv", index=False, encoding="utf-8-sig")

print("Saved: combination_terms_onehot.csv")