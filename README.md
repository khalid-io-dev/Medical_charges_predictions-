# 🧠 Health Insurance Cost Predictor

## 📘 Project Overview
This project is developed as part of a health insurance initiative to build an **intelligent system** capable of accurately **predicting medical charges** for insured individuals.  

The main goal is to **anticipate healthcare costs**, **improve transparency** for clients, and **help the company optimize pricing policies**.  

The dataset used includes key attributes such as:
- **Age**
- **Sex**
- **Body Mass Index (BMI)**
- **Number of dependent children**
- **Smoking habit**
- **Geographical region**

---

## 🗂️ Project Structure

```bash
project/
├── data/
│ └── assurance-maladie.csv # Dataset file
├── src/
│ ├── data_preprocessing.py # Data cleaning and preprocessing
│ ├── visualizations.py # EDA and data visualization
│ ├── model_training.py # Model training and evaluation
│ ├── hyperparameter_tuning.py # Model optimization
│ ├── predict.py # Prediction script for new data
│ └── utils.py # Utility functions
├── models/
│ ├── feature_names.joblib
│ ├── scaler.joblib
│ ├── best_model.joblib
│ ├── final_model.joblib
│ ├── rf_best_params.joblib
│ └── xgb_best_params.joblib
├── main.py # Main script to run the pipeline
└── README.md

```


---

## ⚙️ Installation and Setup

### **1️⃣ Clone the repository**

```bash
git clone https://github.com/khalid-io-dev/Medical_charges_predictions-
cd Medical_charges_predictions-
```

 ---


Make sure you have Python 3.8+ installed, then run:

```bash
pip install -r requirements.txt
```

## 3️⃣ Run the main script

```bash
python main.py
```

## 🧠 Technologies Used

Python 3.x

Pandas – Data manipulation

NumPy – Numerical operations

Matplotlib / Seaborn – Data visualization

Scikit-learn – Machine learning models & preprocessing

XGBoost – Gradient boosting model

Joblib – Model persistence

## 📈 Evaluation Metrics

Each model is evaluated using:

RMSE (Root Mean Squared Error)

MAE (Mean Absolute Error)

R² (Coefficient of Determination)

## 🧾 Expected Outcomes

A trained and optimized model capable of predicting medical costs based on user input

Visual analytics to interpret relationships between features and charges

A simple interface to make predictions in real time
