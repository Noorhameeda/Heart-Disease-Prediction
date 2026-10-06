# ============================================================
# HEART DISEASE PREDICTION USING SVM
# CROSS-VALIDATION
# ============================================================

import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("dataset/heart.csv")

print("=" * 70)
print("HEART DISEASE PREDICTION USING SVM")
print("5-FOLD CROSS-VALIDATION")
print("=" * 70)

print("\nDataset Shape:", df.shape)

# ============================================================
# 2. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop("target", axis=1)
y = df["target"]

# ============================================================
# 3. DEFINE MODELS
# ============================================================

models = {
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
    )
}

# ============================================================
# 4. CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\n" + "=" * 70)
print("CROSS-VALIDATION RESULTS")
print("=" * 70)

results = []

for name, model in models.items():

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", model)
    ])

    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )

    mean_score = scores.mean()
    std_score = scores.std()

    results.append({
        "Model": name,
        "Mean Accuracy": mean_score,
        "Standard Deviation": std_score
    })

    print("\nModel:", name)

    print("Fold 1:", round(scores[0] * 100, 2), "%")
    print("Fold 2:", round(scores[1] * 100, 2), "%")
    print("Fold 3:", round(scores[2] * 100, 2), "%")
    print("Fold 4:", round(scores[3] * 100, 2), "%")
    print("Fold 5:", round(scores[4] * 100, 2), "%")

    print("Mean Accuracy:", round(mean_score * 100, 2), "%")
    print("Standard Deviation:", round(std_score * 100, 2), "%")

# ============================================================
# 5. FINAL RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("FINAL CROSS-VALIDATION COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Mean Accuracy": "{:.4f}".format,
            "Standard Deviation": "{:.4f}".format
        }
    )
)

# ============================================================
# 6. SAVE RESULTS
# ============================================================

results_df.to_csv(
    "model/svm_cross_validation.csv",
    index=False
)

print("\nResults saved successfully!")
print("File: model/svm_cross_validation.csv")

print("\n" + "=" * 70)
print("CROSS-VALIDATION COMPLETED!")
print("=" * 70)