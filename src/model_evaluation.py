import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("=" * 70)
print("PHASE 11 - MODEL EVALUATION")
print("=" * 70)

# Load data
X_test = pd.read_csv("output/X_test.csv")
y_test = pd.read_csv("output/y_test.csv").squeeze()

# Load trained model
model = joblib.load("models/random_forest_model.pkl")

# Prediction
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nEvaluation Metrics")
print("-" * 40)
print(f"MAE  : {mae:.6f}")
print(f"MSE  : {mse:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"R²   : {r2:.6f}")

# ---------------------------------------------------
# Actual vs Predicted
# ---------------------------------------------------

plt.figure(figsize=(7,6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.7
)

plt.xlabel("Actual SOH")
plt.ylabel("Predicted SOH")
plt.title("Actual vs Predicted SOH")

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red"
)

plt.tight_layout()
plt.savefig("output/actual_vs_predicted.png")
plt.close()

print("\nActual vs Predicted graph saved.")

# ---------------------------------------------------
# Feature Importance
# ---------------------------------------------------

importance = pd.Series(
    model.feature_importances_,
    index=X_test.columns
).sort_values(ascending=True)

plt.figure(figsize=(7,5))

importance.plot(kind="barh")

plt.title("Feature Importance")
plt.xlabel("Importance")

plt.tight_layout()
plt.savefig("output/feature_importance.png")
plt.close()

print("Feature Importance graph saved.")

print("=" * 70)
print("PHASE 11 COMPLETED")
print("=" * 70)