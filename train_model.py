# ============================================================
# HEART DISEASE PREDICTION USING SVM
# STEP 17 - MODEL TRAINING
# ============================================================

# Import required libraries
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("dataset/heart.csv")

print("=" * 60)
print("HEART DISEASE PREDICTION USING SVM")
print("=" * 60)

print("\nDataset loaded successfully!")

print("Dataset Shape:", df.shape)


# ============================================================
# 2. SEPARATE FEATURES AND TARGET
# ============================================================

# X contains the input features
X = df.drop("target", axis=1)

# y contains the output/target
y = df["target"]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)


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

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# ============================================================
# 4. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed!")


# ============================================================
# 5. CREATE SVM MODEL
# ============================================================

model = SVC(
    kernel="rbf",
    C=100,
    gamma="scale",
    random_state=42
)

print("\nSVM model created!")
print("Kernel: RBF")


# ============================================================
# 6. TRAIN THE SVM MODEL
# ============================================================

model.fit(X_train_scaled, y_train)

print("\nSVM model trained successfully!")


# ============================================================
# 7. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test_scaled)

print("Predictions completed!")


# ============================================================
# 8. CALCULATE ACCURACY
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print("\nAccuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100)


# ============================================================
# 9. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ============================================================
# 11. SAVE MODEL AND SCALER
# ============================================================

joblib.dump(model, "model/heart_svm_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("\nModel saved successfully!")
print("Saved file: model/heart_svm_model.pkl")

print("Scaler saved successfully!")
print("Saved file: model/scaler.pkl")


# ============================================================
# END
# ============================================================

print("\n" + "=" * 60)
print("SVM TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)