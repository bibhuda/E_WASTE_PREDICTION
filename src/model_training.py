# ------------------------------------------
# MODEL TRAINING + VISUALIZATION
# ------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

print("Loading selected features dataset...")

df = pd.read_csv("data/processed/selected_features_dataset.csv")

# ------------------------------------------
# Define Features & Target
# ------------------------------------------

X = df[["GNI_per_capita", "Internet_Users"]]
y = df["E_Waste_per_capita"]

# ------------------------------------------
# Train-Test Split
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------
# Train Model
# ------------------------------------------

model = LinearRegression()
model.fit(X_train, y_train)

# ------------------------------------------
# Predictions
# ------------------------------------------

y_pred = model.predict(X_test)

# ------------------------------------------
# Evaluation
# ------------------------------------------

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\nModel Performance:")
print("R² Score:", round(r2, 4))
print("RMSE:", round(rmse, 4))

# ------------------------------------------
# Visualization (Single Plot)
# ------------------------------------------

plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual E-Waste per Capita")
plt.ylabel("Predicted E-Waste per Capita")
plt.title("Actual vs Predicted E-Waste per Capita")
plt.show()

# ------------------------------------------
# Save Metrics
# ------------------------------------------

with open("results/model_metrics.txt", "w") as f:
    f.write(f"R2 Score: {r2}\n")
    f.write(f"RMSE: {rmse}\n")

print("\nModel metrics saved in results folder.")