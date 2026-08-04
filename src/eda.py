import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/dataset.csv")

# Set plot style
sns.set_style("whitegrid")

print("Dataset Shape:", df.shape)
print(df.head())


# SOH Distribution
plt.figure(figsize=(8, 5))

sns.histplot(df["soh"], bins=20, kde=True)

plt.title("State of Health (SOH) Distribution")
plt.xlabel("SOH")
plt.ylabel("Count")

plt.show()


# RUL Distribution
plt.figure(figsize=(8, 5))

sns.histplot(df["rul"], bins=20, color="green", kde=True)

plt.title("Remaining Useful Life (RUL) Distribution")
plt.xlabel("RUL")
plt.ylabel("Count")

plt.show()

#voltage plot
plt.figure(figsize=(8,5))

sns.histplot(df["voltage"], bins=20, color="orange", kde=True)

plt.title("Voltage Distribution")
plt.xlabel("Voltage (V)")
plt.ylabel("Count")

plt.show()


#temperature plot
plt.figure(figsize=(8,5))

sns.histplot(df["temperature"], bins=20, color="red", kde=True)

plt.title("Temperature Distribution")
plt.xlabel("Temperature (°C)")
plt.ylabel("Count")

plt.show()