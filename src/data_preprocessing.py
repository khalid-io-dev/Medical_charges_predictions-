import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib

def load_and_clean_data(file_path='data/assurance-maladie.csv'):
    """Load and clean the dataset."""
    df = pd.read_csv(file_path)
    
    print("Duplicate rows:", df.duplicated().sum())
    print("Null values:\n", df.isnull().sum())
    
    df = df.drop_duplicates()
    
    df.info()
    return df

def feature_engineering(df):
    """Perform feature engineering."""
    df_encoded = df.copy()
    
    df_encoded["has_children"] = (df["children"] > 0).astype(int)
    df_encoded["age_smoker"] = df["age"] * (df["smoker"] == "yes").astype(int)
    df_encoded["bmi_smoker"] = df["bmi"] * (df["smoker"] == "yes").astype(int)
    df_encoded["smoker_region"] = df["smoker"] + "_" + df["region"]
    
    df_encoded = pd.get_dummies(df_encoded, columns=["sex", "smoker_region", "region"], drop_first=True)
    
    return df_encoded

def prepare_data(df_encoded, target='charges', test_size=0.2, random_state=42):
    """Split and scale data."""
    X = df_encoded.drop(target, axis=1)
    y = df_encoded[target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Handle non-numeric columns
    if 'smoker' in X_train.columns:
        X_train['smoker_yes'] = (X_train['smoker'] == 'yes').astype(float)
        X_test['smoker_yes'] = (X_test['smoker'] == 'yes').astype(float)
        X_train = X_train.drop(columns=['smoker'])
        X_test = X_test.drop(columns=['smoker'])
    
    bool_cols = X_train.select_dtypes(include=['bool']).columns.tolist()
    if bool_cols:
        X_train[bool_cols] = X_train[bool_cols].astype(float)
        X_test[bool_cols] = X_test[bool_cols].astype(float)
    
    feature_names = X_train.columns.tolist()
    joblib.dump(feature_names, "models/feature_names.joblib")
    
    X_train_unscaled = X_train.copy()
    X_test_unscaled = X_test.copy()
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    joblib.dump(scaler, "models/scaler.joblib")
    
    return (X_train_scaled, X_test_scaled, X_train_unscaled, X_test_unscaled,
            y_train, y_test, feature_names, scaler)

def prepare_robust_scaled_data(X_train_unscaled, X_test_unscaled, y_train, y_test):
    """Prepare data with RobustScaler."""
    scaler = RobustScaler()
    X_train_rob_scaled = scaler.fit_transform(X_train_unscaled)
    X_test_rob_scaled = scaler.transform(X_test_unscaled)
    return X_train_rob_scaled, X_test_rob_scaled, scaler