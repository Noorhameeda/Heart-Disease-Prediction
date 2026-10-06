# ============================================================
# HEART DISEASE PREDICTION USING SVM
# STEP 10 - DATA EXPLORATION
# ============================================================

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("dataset/heart.csv")

print("=" * 60)
print("HEART DISEASE DATASET")
print("=" * 60)

print("\nDataset loaded successfully!")


# ============================================================
# 2. DISPLAY FIRST 5 ROWS
# ============================================================

print("\nFirst 5 rows of the dataset:")
print(df.head())


# ============================================================
# 3. CHECK DATASET SHAPE
# ============================================================

print("\nDataset Shape:")
print(df.shape)


# ============================================================
# 4. DISPLAY COLUMN NAMES
# ============================================================

print("\nColumn Names:")
print(df.columns.tolist())


# ============================================================
# 5. CHECK MISSING VALUES
# ============================================================

print("\nMissing Values in Each Column:")
print(df.isnull().sum())

print("\nTotal Missing Values:")
print(df.isnull().sum().sum())


# ============================================================
# 6. DATASET INFORMATION
# ============================================================

print("\nDataset Information:")
df.info()


# ============================================================
# 7. TARGET DISTRIBUTION
# ============================================================

print("\nTarget Distribution:")
print(df["target"].value_counts())


# ============================================================
# 8. BASIC STATISTICS
# ============================================================

print("\nBasic Statistical Information:")
print(df.describe())


# ============================================================
# 9. AGE DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    df["age"],
    bins=20,
    kde=True
)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Patients")

plt.tight_layout()
plt.show()


# ============================================================
# 10. TARGET DISTRIBUTION GRAPH
# ============================================================

plt.figure(figsize=(7, 5))

sns.countplot(
    x="target",
    data=df
)

plt.title("Heart Disease Target Distribution")
plt.xlabel("Target")
plt.ylabel("Number of Patients")

plt.tight_layout()
plt.show()


# ============================================================
# 11. CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(12, 8))

sns.heatmap(
    df.corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.show()


# ============================================================
# END
# ============================================================

print("\n" + "=" * 60)
print("DATA EXPLORATION COMPLETED SUCCESSFULLY!")
print("=" * 60)