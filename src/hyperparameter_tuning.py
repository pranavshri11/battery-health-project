import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

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
print("RANDOM FOREST HYPERPARAMETER TUNING")
print("=" * 70)

# =====================================================
# Base Model
# =====================================================

rf = RandomForestRegressor(random_state=42)

# =====================================================
# Parameter Grid
# =====================================================

param_grid = {

    "n_estimators": [100, 200],

    "max_depth": [None, 10, 20],

    "min_samples_split": [2, 5],

    "min_samples_leaf": [1, 2],

    "max_features": ["sqrt", "log2"]

}

# =====================================================
# Grid Search
# =====================================================

grid_search = GridSearchCV(

    estimator=rf,

    param_grid=param_grid,

    cv=5,

    scoring="r2",

    n_jobs=-1,

    verbose=2

)

print("\nTraining... Please wait.\n")

grid_search.fit(X_train, y_train)

# =====================================================
# Best Model
# =====================================================

best_model = grid_search.best_estimator_

prediction = best_model.predict(X_test)

# =====================================================
# Evaluation
# =====================================================

mae = mean_absolute_error(y_test, prediction)

mse = mean_squared_error(y_test, prediction)

rmse = mse ** 0.5

r2 = r2_score(y_test, prediction)

# =====================================================
# Results
# =====================================================

print("\nBest Parameters")

print(grid_search.best_params_)

print("\nEvaluation")

print(f"MAE  : {mae:.6f}")

print(f"MSE  : {mse:.6f}")

print(f"RMSE : {rmse:.6f}")

print(f"R²   : {r2:.6f}")

# =====================================================
# Save Model
# =====================================================

os.makedirs("models", exist_ok=True)

joblib.dump(

    best_model,

    "models/random_forest_model.pkl"

)

print("\nModel Saved")

print("models/random_forest_model.pkl")

print("=" * 70)
print("PHASE 10 COMPLETED")
print("=" * 70)