import pandas as pd

df = pd.read_csv("data/dataset.csv")

print("=" * 60)
print("DATA QUALITY REPORT")
print("=" * 60)

print("\n1. Missing Values")
print(df.isnull().sum())

print("\n2. Duplicate Rows")
print(df.duplicated().sum())

print("\n3. Data Types")
print(df.dtypes)

print("\n4. Unique Batteries")
print(df["battery_id"].nunique())

print("\n5. Records Per Battery")
print(df["battery_id"].value_counts())

print("\n6. Unique Values")
print(df.nunique())

print("\n7. Memory Usage")
print(df.memory_usage(deep=True))

print("\n8. Dataset Shape")
print(df.shape)