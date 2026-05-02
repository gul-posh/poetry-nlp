import pandas as pd

# Load both datasets
df_full = pd.read_csv("data/shakespeare_full.csv")
df_small = pd.read_csv("data/poems.csv")

# Keep only relevant columns from small dataset
df_small = df_small[["sonnet_number", "label"]]

# Merge
df_merged = df_full.merge(df_small, on="sonnet_number", how="left")

df_merged.to_csv("data/poems_extended.csv", index=False)

print("Merged dataset created.")
print(df_merged.head())
print("\nLabeled count:", df_merged["label"].notna().sum())
