import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------

df = pd.read_csv("data/dataset.csv")

print("=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

# ---------------------------------------------------
# Features and Target
# ---------------------------------------------------

X = df[[
    "cycle",
    "voltage",
    "temperature",
    "capacity"
]]

y = df["soh"]

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print(y.name)

# ---------------------------------------------------
# Train Test Split
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape :", X_test.shape)

# ---------------------------------------------------
# Feature Scaling
# ---------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nScaling Completed")

# ---------------------------------------------------
# Convert back to DataFrame
# ---------------------------------------------------

X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=X.columns
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=X.columns
)

print("\nScaled Training Data")
print(X_train_scaled.head())

# ---------------------------------------------------
# Save Processed Data
# ---------------------------------------------------

X_train_scaled.to_csv("output/X_train.csv", index=False)
X_test_scaled.to_csv("output/X_test.csv", index=False)

y_train.to_csv("output/y_train.csv", index=False)
y_test.to_csv("output/y_test.csv", index=False)

print("\nProcessed datasets saved inside output/")

print("=" * 60)
print("PHASE 8 COMPLETED")
print("=" * 60)


import joblib
import os

os.makedirs("models", exist_ok=True)

joblib.dump(scaler, "models/scaler.pkl")