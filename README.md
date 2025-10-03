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



---

## 🧩 Feature Stories

### **Feature Story 1: Data Analysis and Preparation**
**Objective:** Ensure the dataset is clean, consistent, and ready for model training.  

**Tasks:**
1. **Load Data**
   - Import data using Pandas  
   - Inspect column types and structure  

2. **Exploratory Data Analysis (EDA)**
   - Generate descriptive statistics  
   - Identify missing values and duplicates  
   - Visualize distributions (histograms, pairplots, heatmaps)  
   - Explore correlations among variables  

3. **Data Preprocessing**
   - Handle missing values (median/mode)
   - Remove duplicates  
   - Detect and treat outliers (IQR or z-score)  
   - Encode categorical variables (one-hot or label encoding)  
   - Split data into training/test sets (80%/20%)  
   - Apply normalization or standardization (MinMaxScaler / StandardScaler)  

---

### **Feature Story 2: Training Regression Models**
**Objective:** Train multiple regression models and evaluate their initial performance.  

**Models:**
- Linear Regression  
- Random Forest Regressor  
- XGBoost Regressor  
- Support Vector Regressor (SVR)

**Tasks:**
- Train models with default hyperparameters  
- Integrate preprocessing and training using Scikit-learn Pipelines  
- Evaluate performance (RMSE, MAE, R²)  
- Record results for comparison  

---

### **Feature Story 3: Hyperparameter Tuning**
**Objective:** Optimize the best-performing models to improve prediction accuracy.  

**Tasks:**
- Select top models from previous step  
- Perform **GridSearchCV** or **RandomizedSearchCV** with cross-validation (5 folds)  
- Tune parameters (e.g., `n_estimators`, `max_depth`, `learning_rate`, `min_samples_split`)  
- Compare pre- and post-tuning performance  
- Retrain optimized models on full training data  

---

### **Feature Story 4: Model Evaluation and Comparison**
**Objective:** Evaluate all optimized models and choose the final one.  

**Tasks:**
- Create comparison visualizations (residual plots, prediction vs. actual)  
- Build a summary table (RMSE, MAE, R²) for all models  
- Select the final model based on performance and stability  

---

### **Feature Story 5: Model Testing and Deployment**
**Objective:** Test the final model and provide a simple interface for predictions.  

**Tasks:**
- Export final model (`joblib.dump()` or `pickle`)  
- Create an interface (CLI or web form) to input user data  
- Use the trained model to predict estimated medical charges  

---

## ⚙️ Installation and Setup

### **1️⃣ Clone the repository**
```bash
git clone https://github.com/khalid-io-dev/Medical_charges_predictions-
cd Medical_charges_predictions-

 ---


Make sure you have Python 3.8+ installed, then run:

pip install -r requirements.txt

3️⃣ Run the main script

python main.py

🧠 Technologies Used

Python 3.x

Pandas – Data manipulation

NumPy – Numerical operations

Matplotlib / Seaborn – Data visualization

Scikit-learn – Machine learning models & preprocessing

XGBoost – Gradient boosting model

Joblib – Model persistence

📈 Evaluation Metrics

Each model is evaluated using:

RMSE (Root Mean Squared Error)

MAE (Mean Absolute Error)

R² (Coefficient of Determination)

🧾 Expected Outcomes

A trained and optimized model capable of predicting medical costs based on user input

Visual analytics to interpret relationships between features and charges

A simple interface to make predictions in real time
