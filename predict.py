# ============================================================
# HEART DISEASE PREDICTION USING SVM
# DATASET-BASED PREDICTION TEST
# ============================================================

import pandas as pd
import joblib

# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("dataset/heart.csv")

# ============================================================
# 2. LOAD TRAINED MODEL AND SCALER
# ============================================================

model = joblib.load("model/heart_svm_model.pkl")
scaler = joblib.load("model/scaler.pkl")

print("=" * 60)
print("HEART DISEASE PREDICTION USING SVM")
print("=" * 60)

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)

# ============================================================
# 3. SELECT A PATIENT FROM DATASET
# ============================================================

patient_number = 0

patient = df.iloc[patient_number]

# Get the input features
X_patient = patient.drop("target")

# Get the actual target
actual_target = patient["target"]

# Convert to DataFrame
patient_data = pd.DataFrame(
    [X_patient.values],
    columns=X_patient.index
)

# ============================================================
# 4. DISPLAY PATIENT INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("PATIENT DATA")
print("=" * 60)

print("\nPatient Record Number:", patient_number + 1)

print("\nPatient Features:")

for column in patient_data.columns:
    print(column, ":", patient_data.iloc[0][column])

# ============================================================
# 5. SCALE PATIENT DATA
# ============================================================

patient_scaled = scaler.transform(patient_data)

# ============================================================
# 6. MAKE PREDICTION
# ============================================================

prediction = model.predict(patient_scaled)

predicted_target = prediction[0]

# ============================================================
# 7. DISPLAY ACTUAL AND PREDICTED RESULTS
# ============================================================

print("\n" + "=" * 60)
print("PREDICTION RESULT")
print("=" * 60)

print("\nActual Target:", int(actual_target))
print("Predicted Target:", int(predicted_target))

if predicted_target == 1:
    print("\nPrediction: Heart Disease Detected")
else:
    print("\nPrediction: No Heart Disease Detected")

# ============================================================
# 8. CHECK WHETHER PREDICTION IS CORRECT
# ============================================================

if predicted_target == actual_target:
    print("\nPrediction Status: CORRECT")
else:
    print("\nPrediction Status: INCORRECT")

print("\nNote: This is a machine-learning prediction and")
print("should not be considered a medical diagnosis.")

print("\n" + "=" * 60)
print("PREDICTION TEST COMPLETED!")
print("=" * 60)