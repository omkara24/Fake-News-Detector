import pandas as pd

# Load fake and real news datasets
fake_df = pd.read_csv("data/fake.csv")
true_df = pd.read_csv("data/true.csv")

# Add labels
fake_df["label"] = "FAKE"
true_df["label"] = "REAL"

# Combine both datasets
df = pd.concat([fake_df, true_df], axis=0)

# Shuffle data
df = df.sample(frac=1).reset_index(drop=True)

# Basic information
print("First 5 rows:\n")
print(df.head())

print("\nDataset Info:\n")
print(df.info())

print("\nClass Distribution:\n")
print(df["label"].value_counts())

print("\nTotal Records:", len(df))

