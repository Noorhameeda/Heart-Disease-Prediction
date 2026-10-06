import streamlit as st
import pandas as pd
import joblib

# ============================================================
# HEART DISEASE PREDICTION USING SVM
# Streamlit Web Application
# ============================================================

# Load trained model and scaler
model = joblib.load("model/heart_svm_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# ============================================================
# TITLE
# ============================================================

st.title("❤️ Heart Disease Prediction")
st.subheader("Machine Learning Prediction using Support Vector Machine (SVM)")

st.write(
    "Enter the patient's medical information below to predict "
    "the possibility of heart disease."
)

st.divider()

# ============================================================
# PATIENT INFORMATION
# ============================================================

st.header("👤 Patient Information")

st.info(
    "Please enter the patient's information carefully. "
    "The descriptions below explain what each feature represents."
)

col1, col2 = st.columns(2)

# ============================================================
# LEFT COLUMN
# ============================================================

with col1:

    st.subheader("Basic Information")

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=50
    )

    st.caption("Age of the patient in years.")

    sex = st.selectbox(
        "Sex",
        options=[0, 1],
        format_func=lambda x: "Female" if x == 0 else "Male"
    )

    st.caption("Sex of the patient: Female = 0, Male = 1.")

    cp = st.selectbox(
        "Chest Pain Type",
        options=[0, 1, 2, 3]
    )

    st.caption(
        "Type of chest pain recorded for the patient. "
        "Values 0–3 represent different chest-pain categories."
    )

    trestbps = st.number_input(
        "Resting Blood Pressure",
        min_value=50,
        max_value=250,
        value=120
    )

    st.caption(
        "Resting blood pressure measured in mm Hg."
    )

    chol = st.number_input(
        "Cholesterol",
        min_value=50,
        max_value=700,
        value=200
    )

    st.caption(
        "Serum cholesterol level measured in mg/dl."
    )

    fbs = st.selectbox(
        "Fasting Blood Sugar",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    st.caption(
        "Whether fasting blood sugar is greater than 120 mg/dl. "
        "No = 0, Yes = 1."
    )

    restecg = st.selectbox(
        "Resting ECG",
        options=[0, 1, 2]
    )

    st.caption(
        "Result of the resting electrocardiogram (ECG). "
        "Values 0–2 represent different ECG categories."
    )


# ============================================================
# RIGHT COLUMN
# ============================================================

with col2:

    st.subheader("Heart & Exercise Information")

    thalach = st.number_input(
        "Maximum Heart Rate",
        min_value=50,
        max_value=250,
        value=150
    )

    st.caption(
        "Maximum heart rate achieved during the test."
    )

    exang = st.selectbox(
        "Exercise Induced Angina",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    st.caption(
        "Whether exercise causes angina (chest discomfort). "
        "No = 0, Yes = 1."
    )

    oldpeak = st.number_input(
        "ST Depression (Oldpeak)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

    st.caption(
        "ST-segment depression measured during exercise "
        "relative to rest."
    )

    slope = st.selectbox(
        "Slope",
        options=[0, 1, 2]
    )

    st.caption(
        "Slope of the peak exercise ST segment. "
        "Values 0–2 represent different slope categories."
    )

    ca = st.selectbox(
        "Number of Major Vessels",
        options=[0, 1, 2, 3, 4]
    )

    st.caption(
        "Number of major vessels recorded by fluoroscopy."
    )

    thal = st.selectbox(
        "Thal",
        options=[0, 1, 2, 3]
    )

    st.caption(
        "Thalassemia-related categorical value used in the dataset."
    )


st.divider()

# ============================================================
# PREDICTION BUTTON
# ============================================================

predict_button = st.button(
    "🔍 Predict Heart Disease",
    type="primary",
    use_container_width=True
)

# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    patient_data = pd.DataFrame(
        [[
            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal
        ]],
        columns=[
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal"
        ]
    )

    # Scale patient input
    patient_scaled = scaler.transform(patient_data)

    # Make prediction
    prediction = model.predict(patient_scaled)

    # ========================================================
    # PREDICTION RESULT
    # ========================================================

    st.divider()

    st.header("📊 Prediction Result")

    if prediction[0] == 1:

        st.error(
            """
            ## ❤️ Heart Disease Detected

            The model predicts a positive heart-disease result
            based on the information entered.
            """,
            icon="❤️"
        )

        st.warning(
            "Please consult a qualified healthcare professional "
            "for proper medical evaluation."
        )

    else:

        st.success(
            """
            ## ✅ No Heart Disease Detected

            The model predicts a negative heart-disease result
            based on the information entered.
            """,
            icon="✅"
        )

        st.info(
            "This prediction is based on the machine-learning model "
            "and does not replace professional medical evaluation."
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

st.header("🤖 Model Information")

info_col1, info_col2, info_col3, info_col4 = st.columns(4)

with info_col1:
    st.metric(
        "Algorithm",
        "SVM"
    )

with info_col2:
    st.metric(
        "Kernel",
        "RBF"
    )

with info_col3:
    st.metric(
        "C Value",
        "100"
    )

with info_col4:
    st.metric(
        "5-Fold CV Accuracy",
        "99.61%"
    )

st.write(
    "The model was trained on 1,025 records using 13 input features. "
    "The reported 99.61% value is the mean accuracy obtained from "
    "5-fold cross-validation on this dataset."
)


# ============================================================
# MEDICAL DISCLAIMER
# ============================================================

st.divider()

st.warning(
    """
    ⚠️ **Medical Disclaimer**

    This application provides a machine-learning prediction based
    on the entered data. It is intended for educational and
    demonstration purposes only and should not be considered a
    medical diagnosis or a substitute for professional medical advice.
    """
)