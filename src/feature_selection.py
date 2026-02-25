# ------------------------------------------
# FEATURE SELECTION SCRIPT
# ------------------------------------------

import pandas as pd

print("Loading final dataset...")

df = pd.read_csv("data/processed/final_dataset.csv")

print("Dataset Loaded!")
print("Shape:", df.shape)

# ------------------------------------------
# 1️⃣ Correlation Matrix
# ------------------------------------------

print("\nCorrelation Matrix:\n")
correlation = df.corr(numeric_only=True)
print(correlation)

# ------------------------------------------
# 2️⃣ Select Relevant Features
# ------------------------------------------

# We choose features based on correlation strength
features = ["GNI_per_capita", "Internet_Users"]

X = df[features]
y = df["E_Waste_per_capita"]

print("\nSelected Features:")
print(features)

# ------------------------------------------
# 3️⃣ Save Selected Dataset
# ------------------------------------------

selected_df = df[["Country", "Year"] + features + ["E_Waste_per_capita"]]
selected_df.to_csv("data/processed/selected_features_dataset.csv", index=False)

print("\nSelected features dataset saved successfully!")
print("New Shape:", selected_df.shape)