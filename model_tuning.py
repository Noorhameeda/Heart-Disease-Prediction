# ============================================================
# HEART DISEASE PREDICTION USING SVM
# MODEL TUNING AND COMPARISON
# ============================================================

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("dataset/heart.csv")

print("=" * 70)
print("HEART DISEASE PREDICTION USING SVM")
print("MODEL TUNING AND COMPARISON")
print("=" * 70)

print("\nDataset Shape:", df.shape)

# ============================================================
# 2. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop("target", axis=1)
y = df["target"]

# ============================================================
# 3. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================================================
# 4. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed!")

# ============================================================
# 5. DEFINE SVM MODELS
# ============================================================

models = {
    "Linear SVM": SVC(
        kernel="linear",
        C=1
    ),

    "RBF SVM": SVC(
        kernel="rbf",
        C=1,
        gamma="scale"
    ),

    "RBF SVM - C=10": SVC(
        kernel="rbf",
        C=10,
        gamma="scale"
    ),

    "RBF SVM - C=100": SVC(
        kernel="rbf",
        C=100,
        gamma="scale"
    ),

    "Polynomial SVM": SVC(
        kernel="poly",
        C=1,
        degree=3
    )
}

# ============================================================
# 6. TRAIN AND EVALUATE MODELS
# ============================================================

results = []

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

for name, model in models.items():

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print("\nModel:", name)
    print("Accuracy :", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision * 100, 2), "%")
    print("Recall   :", round(recall * 100, 2), "%")
    print("F1 Score :", round(f1 * 100, 2), "%")

# ============================================================
# 7. CREATE RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1 Score": "{:.4f}".format
        }
    )
)

# ============================================================
# 8. SAVE RESULTS
# ============================================================

results_df.to_csv(
    "model/svm_model_comparison.csv",
    index=False
)

print("\nComparison results saved successfully!")
print("File: model/svm_model_comparison.csv")

print("\n" + "=" * 70)
print("MODEL TUNING COMPLETED!")
print("=" * 70)