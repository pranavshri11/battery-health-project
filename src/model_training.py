import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =====================================================
# Load Data
# =====================================================

X_train = pd.read_csv("output/X_train.csv")
X_test = pd.read_csv("output/X_test.csv")

y_train = pd.read_csv("output/y_train.csv").squeeze()
y_test = pd.read_csv("output/y_test.csv").squeeze()

print("=" * 70)
print("BATTERY HEALTH PREDICTION - MODEL COMPARISON")
print("=" * 70)

# =====================================================
# Models
# =====================================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Ridge Regression":
        Ridge(alpha=1.0),

    "Lasso Regression":
        Lasso(alpha=0.001),

    "Decision Tree":
        DecisionTreeRegressor(random_state=42),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ),

    "Extra Trees":
        ExtraTreesRegressor(
            n_estimators=100,
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            random_state=42
        ),

    "XGBoost":
        XGBRegressor(
            objective="reg:squarederror",
            n_estimators=100,
            random_state=42
        )

}

# =====================================================
# Train Models
# =====================================================

results = []

best_model = None
best_name = ""
best_score = -999

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    mae = mean_absolute_error(y_test, prediction)

    mse = mean_squared_error(y_test, prediction)

    rmse = mse ** 0.5

    r2 = r2_score(y_test, prediction)

    results.append([
        name,
        mae,
        mse,
        rmse,
        r2
    ])

    if r2 > best_score:

        best_score = r2

        best_model = model

        best_name = name

# =====================================================
# Results Table
# =====================================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "MAE",
        "MSE",
        "RMSE",
        "R2 Score"
    ]
)

results_df = results_df.sort_values(
    by="R2 Score",
    ascending=False
)

print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(results_df)

# =====================================================
# Save CSV
# =====================================================

os.makedirs("output", exist_ok=True)

results_df.to_csv(
    "output/model_comparison.csv",
    index=False
)

# =====================================================
# Save Best Model
# =====================================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    best_model,
    "models/best_model.pkl"
)

print("\nBest Model :", best_name)

print("Best R² :", round(best_score,4))

# =====================================================
# Accuracy Graph
# =====================================================

plt.figure(figsize=(12,6))

plt.bar(
    results_df["Model"],
    results_df["R2 Score"]
)

plt.xticks(rotation=25)

plt.ylabel("R² Score")

plt.title("Machine Learning Model Comparison")

plt.tight_layout()

plt.savefig(
    "output/model_comparison.png",
    dpi=300
)

plt.show()

print("\nGraph Saved")

print("output/model_comparison.png")

print("\nBest Model Saved")

print("models/best_model.pkl")

print("=" * 70)
print("PHASE 9 COMPLETED")
print("=" * 70)