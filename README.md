# Heart Disease Prediction Using Support Vector Machine (SVM)

## 📌 Project Overview

Heart Disease Prediction is a machine learning project that uses a **Support Vector Machine (SVM)** algorithm to predict the presence or absence of heart disease based on patient health information.

The project uses Python and Scikit-learn to train and evaluate the model. A user-friendly web interface is developed using Streamlit, allowing users to enter patient information and receive a model prediction.

This project is developed for **educational and demonstration purposes only** and is not intended to provide a medical diagnosis.

## 🎯 Objectives

- Analyze a heart disease dataset.
- Preprocess data and scale numerical features.
- Train a Support Vector Machine classifier.
- Compare different SVM kernels and hyperparameter configurations.
- Evaluate model performance using classification metrics.
- Validate the model using 5-fold cross-validation.
- Develop an interactive web application using Streamlit.

## ✨ Features

- **Data Exploration:** Load and examine the heart disease dataset.
- **Feature Scaling:** Apply `StandardScaler` to standardize input features.
- **SVM Classification:** Use an SVM classifier with an RBF kernel.
- **Model Comparison:** Compare Linear, RBF, and Polynomial SVM configurations.
- **Hyperparameter Tuning:** Evaluate different values of the `C` parameter.
- **Model Evaluation:** Measure accuracy, precision, recall, F1-score, and confusion matrix.
- **Cross-Validation:** Evaluate model performance using stratified 5-fold cross-validation.
- **Interactive Web App:** Enter patient information and view the prediction through Streamlit.
- **Model Persistence:** Save the trained model and scaler using Joblib.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Dataset loading and data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Model training, preprocessing, and evaluation |
| SVM | Classification algorithm |
| StandardScaler | Feature scaling |
| Joblib | Saving and loading the model |
| Streamlit | Web application interface |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Git and GitHub | Version control and project hosting |

## 📂 Project Structure
Heart_Disease_SVM/
│
├── dataset/
│   └── heart.csv
│
├── model/
│   ├── heart_svm_model.pkl
│   ├── scaler.pkl
│   ├── svm_model_comparison.csv
│   └── svm_cross_validation.csv
│
├── src/
│   ├── data_exploration.py
│   ├── train_model.py
│   ├── predict.py
│   ├── model_tuning.py
│   └── model_validation.py
│
├── app.py
├── requirements.txt
└── README.md

## 📊 Dataset Information

The dataset contains **1,025 records and 14 columns**.

- **Input features:** 13
- **Target variable:** 1 (`target`)
- **Task:** Binary classification

The input features include:

| Feature | Description |
|---|---|
| `age` | Age of the patient |
| `sex` | Sex category encoded in the dataset |
| `cp` | Chest pain type |
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol level |
| `fbs` | Fasting blood sugar category |
| `restecg` | Resting electrocardiographic result |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina |
| `oldpeak` | ST depression measurement |
| `slope` | Slope of the peak exercise ST segment |
| `ca` | Number of major vessels |
| `thal` | Thal-related categorical feature |
| `target` | Target class used for training |

**Target interpretation:** In this dataset, `0` represents the negative class and `1` represents the positive class. These are model labels, not a substitute for clinical assessment.

## ⚙️ Installation and Setup

### 1. Clone the Repository

git clone https://github.com/YOUR-USERNAME/Heart_Disease_SVM.git

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Open the Project Folder

cd Heart_Disease_SVM

### 3. Create a Virtual Environment

py -m venv venv

### 4. Activate the Virtual Environment

For Windows PowerShell:
.\venv\Scripts\Activate.ps1

### 5. Install Dependencies
py -m pip install -r requirements.txt

If the `py` command does not work inside the virtual environment, use:

```powershell ```
.\venv\Scripts\python.exe -m pip install -r requirements.txt

## 🚀 How to Run the Project

### Run the Streamlit Web Application

From the project root directory, execute:

py -m streamlit run app.py

The application should open in your browser. If it does not, visit:

http://localhost:8501

Enter the patient information and click **Predict Heart Disease** to view the model's prediction.

### Train the Model

To train the SVM model again:

py src/train_model.py

The script evaluates the model and saves the trained model and scaler in the `model/` directory.

### Explore the Dataset
py src/data_exploration.py

### Test a Prediction

py src/predict.py

### Compare SVM Models

py src/model_tuning.py

### Run 5-Fold Cross-Validation

py src/model_validation.py

Run these commands from the project root directory.

## 🧠 Machine Learning Methodology

The project follows these steps:

1. Load the heart disease dataset using Pandas.
2. Separate the input features from the target variable.
3. Split the dataset into training and testing sets using an 80:20 ratio.
4. Apply `StandardScaler` to scale the features.
5. Train and compare different SVM configurations.
6. Evaluate model performance using classification metrics.
7. Perform stratified 5-fold cross-validation.
8. Save the selected model and scaler using Joblib.
9. Integrate the saved model into the Streamlit web application.
10. Accept patient information and display the model prediction.

### Selected Model

- **Algorithm:** Support Vector Machine (SVM)
- **Kernel:** Radial Basis Function (RBF)
- **C parameter:** 100
- **Feature scaling:** StandardScaler
- **Validation method:** Stratified 5-fold cross-validation

## 📈 Model Performance

The following results were obtained during the project's model evaluation.

| Evaluation | Result |
|---|---:|
| Held-out test accuracy | 100% |
| 5-fold cross-validation mean accuracy | 99.61% |
| Cross-validation standard deviation | 0.78 percentage points |

The model comparison experiment reported the following held-out test accuracies:

| SVM Configuration | Accuracy |
|---|---:|
| Linear SVM, C = 1 | 81.46% |
| RBF SVM, C = 1 | 92.68% |
| RBF SVM, C = 10 | 98.54% |
| RBF SVM, C = 100 | 100% |
| Polynomial SVM, C = 1 | 92.68% |

**Important:** These scores describe performance on the dataset and evaluation procedures used in this project. They do not establish clinical accuracy or guarantee reliable predictions on new patients. The unusually high scores should be investigated further using independent data and checks for duplicate records or data leakage.

## 🔮 Future Enhancements

- Evaluate the model on independent, external datasets.
- Investigate duplicate records and possible data leakage.
- Compare SVM with Logistic Regression, Random Forest, and other classifiers.
- Add more detailed data visualizations.
- Improve the user interface and prediction explanations.
- Explore explainable AI techniques.
- Develop a mobile-friendly version of the application.
- Conduct further validation before considering any real-world healthcare use.


## 👩‍💻 Author

**Noor Hameeda Shaik**

B.Tech – Computer Science and Engineering

