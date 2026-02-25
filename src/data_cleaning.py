# ------------------------------------------
# DATA CLEANING & FINAL MERGE SCRIPT
# ------------------------------------------

import pandas as pd

print("Loading cleaned datasets...")

# 1️⃣ Load datasets from RAW folder
ewaste = pd.read_csv("data/raw/ewaste_2019_2025_cleaned.csv")
gni = pd.read_csv("data/raw/gni_2019_2025_cleaned.csv")
internet = pd.read_csv("data/raw/internet_2019_2025_cleaned.csv")

print("Datasets loaded successfully!")

# ------------------------------------------
# 2️⃣ Ensure Year column is string (important for merge)
# ------------------------------------------

ewaste["Year"] = ewaste["Year"].astype(str)
gni["Year"] = gni["Year"].astype(str)
internet["Year"] = internet["Year"].astype(str)

# ------------------------------------------
# 3️⃣ Clean country names (strip spaces)
# ------------------------------------------

ewaste["Country"] = ewaste["Country"].str.strip()
gni["Country"] = gni["Country"].str.strip()
internet["Country"] = internet["Country"].str.strip()

# ------------------------------------------
# 4️⃣ Merge datasets
# ------------------------------------------

print("Merging datasets...")

merged = pd.merge(ewaste, gni, on=["Country", "Year"], how="inner")
merged = pd.merge(merged, internet, on=["Country", "Year"], how="inner")

# ------------------------------------------
# 5️⃣ Drop missing values
# ------------------------------------------

merged = merged.dropna()

# ------------------------------------------
# 6️⃣ Save final dataset
# ------------------------------------------

merged.to_csv("data/processed/final_dataset.csv", index=False)

print("\nFinal Dataset Created Successfully!")
print("Final Dataset Shape:", merged.shape)
print(merged.head())