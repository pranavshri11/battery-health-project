import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# SETTINGS
# -----------------------------
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

# Create output folder
os.makedirs("output", exist_ok=True)

# Load dataset
df = pd.read_csv("data/dataset.csv")

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 60)

# -----------------------------
# 1. Dataset Preview
# -----------------------------
print("\nFirst 5 Rows:")
print(df.head())

# -----------------------------
# 2. Cycle Distribution
# -----------------------------
plt.figure()
sns.histplot(df["cycle"], bins=20, kde=True)
plt.title("Cycle Distribution")
plt.xlabel("Cycle")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("output/cycle_distribution.png", dpi=300)
plt.show()

# -----------------------------
# 3. Voltage Distribution
# -----------------------------
plt.figure()
sns.histplot(df["voltage"], bins=25, kde=True, color="green")
plt.title("Voltage Distribution")
plt.xlabel("Voltage")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("output/voltage_distribution.png", dpi=300)
plt.show()

# -----------------------------
# 4. Temperature Distribution
# -----------------------------
plt.figure()
sns.histplot(df["temperature"], bins=25, kde=True, color="orange")
plt.title("Temperature Distribution")
plt.xlabel("Temperature")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("output/temperature_distribution.png", dpi=300)
plt.show()

# -----------------------------
# 5. Capacity Distribution
# -----------------------------
plt.figure()
sns.histplot(df["capacity"], bins=25, kde=True, color="purple")
plt.title("Capacity Distribution")
plt.xlabel("Capacity")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("output/capacity_distribution.png", dpi=300)
plt.show()

# -----------------------------
# 6. SOH Distribution
# -----------------------------
plt.figure()
sns.histplot(df["soh"], bins=25, kde=True, color="red")
plt.title("SOH Distribution")
plt.xlabel("State of Health")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("output/soh_distribution.png", dpi=300)
plt.show()

# -----------------------------
# 7. RUL Distribution
# -----------------------------
plt.figure()
sns.histplot(df["rul"], bins=25, kde=True, color="brown")
plt.title("RUL Distribution")
plt.xlabel("Remaining Useful Life")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("output/rul_distribution.png", dpi=300)
plt.show()

# -----------------------------
# 8. Battery Distribution
# -----------------------------
plt.figure(figsize=(12,6))
sns.countplot(
    data=df,
    x="battery_id",
    order=df["battery_id"].value_counts().index
)
plt.title("Records per Battery")
plt.xlabel("Battery ID")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("output/battery_distribution.png", dpi=300)
plt.show()

# -----------------------------
# 9. Correlation Heatmap
# -----------------------------
plt.figure(figsize=(8,6))
numeric_df = df.select_dtypes(include=["number"])

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("output/correlation_heatmap.png", dpi=300)
plt.show()

# -----------------------------
# 10. Pair Plot
# -----------------------------
pair_df = df.drop(columns=["battery_id"])

pair = sns.pairplot(pair_df)
pair.savefig("output/pairplot.png", dpi=300)
plt.show()

# -----------------------------
# 11. Boxplots
# -----------------------------
features = [
    "voltage",
    "temperature",
    "capacity",
    "soh",
    "rul"
]

for feature in features:
    plt.figure()
    sns.boxplot(x=df[feature])
    plt.title(f"{feature.capitalize()} Boxplot")
    plt.tight_layout()
    plt.savefig(f"output/{feature}_boxplot.png", dpi=300)
    plt.show()

# -----------------------------
# 12. Cycle vs SOH
# -----------------------------
plt.figure(figsize=(10,6))
sns.scatterplot(
    data=df,
    x="cycle",
    y="soh"
)
plt.title("Cycle vs State of Health")
plt.xlabel("Cycle")
plt.ylabel("SOH")
plt.tight_layout()
plt.savefig("output/cycle_vs_soh.png", dpi=300)
plt.show()

# -----------------------------
# 13. Cycle vs Capacity
# -----------------------------
plt.figure(figsize=(10,6))
sns.scatterplot(
    data=df,
    x="cycle",
    y="capacity"
)
plt.title("Cycle vs Capacity")
plt.xlabel("Cycle")
plt.ylabel("Capacity")
plt.tight_layout()
plt.savefig("output/cycle_vs_capacity.png", dpi=300)
plt.show()

# -----------------------------
# 14. Temperature vs SOH
# -----------------------------
plt.figure(figsize=(10,6))
sns.scatterplot(
    data=df,
    x="temperature",
    y="soh"
)
plt.title("Temperature vs SOH")
plt.xlabel("Temperature")
plt.ylabel("SOH")
plt.tight_layout()
plt.savefig("output/temperature_vs_soh.png", dpi=300)
plt.show()

print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY!")
print("All graphs have been saved in the 'output' folder.")
print("=" * 60)