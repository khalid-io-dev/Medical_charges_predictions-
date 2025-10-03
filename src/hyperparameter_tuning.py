from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import make_scorer, mean_squared_error
from xgboost import XGBRegressor
import numpy as np
import joblib

def tune_models(X_train_unscaled, X_train_scaled, y_train):
    """Tune RandomForest, XGBoost, and SVR models."""
    scorer = "neg_mean_squared_error"
    
    # Random Forest
    rf = RandomForestRegressor(random_state=42)
    rf_param_grid = {
        "n_estimators": [100, 200, 400],
        "max_depth": [None, 6, 12, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }
    rf_search = RandomizedSearchCV(
        estimator=rf, param_distributions=rf_param_grid, n_iter=20, scoring=scorer,
        cv=5, verbose=2, random_state=42, n_jobs=-1
    )
    
    # XGBoost
    xgb = XGBRegressor(random_state=42, eval_metric='rmse')
    xgb_param_grid = {
        "n_estimators": [100, 200, 400],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 6, 10],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0]
    }
    xgb_search = RandomizedSearchCV(
        estimator=xgb, param_distributions=xgb_param_grid, n_iter=20, scoring=scorer,
        cv=5, verbose=2, random_state=42, n_jobs=-1
    )
    
    # SVR
    svr = SVR()
    svr_param_grid = {
        "C": [0.1, 1, 10, 100, 1000],
        "epsilon": [0.01, 0.1, 0.5, 1.0],
        "kernel": ['rbf', 'linear', 'poly'],
        "gamma": ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
        "degree": [2, 3, 4]
    }
    svr_search = RandomizedSearchCV(
        estimator=svr, param_distributions=svr_param_grid, n_iter=30,
        scoring=scorer, cv=5, verbose=2, random_state=42, n_jobs=-1
    )

    
    # Fit searches
    print("Running RF RandomizedSearchCV...")
    rf_search.fit(X_train_unscaled, y_train)
    print("Running XGB RandomizedSearchCV...")
    xgb_search.fit(X_train_unscaled, y_train)
    print("Running SVR RandomizedSearchCV...")
    svr_search.fit(X_train_scaled, y_train)  
    
    joblib.dump(rf_search.best_params_, "models/rf_best_params.joblib")
    joblib.dump(xgb_search.best_params_, "models/xgb_best_params.joblib")
    joblib.dump(svr_search.best_params_, "models/svr_best_params.joblib")
    
    rf_best_rmse = np.sqrt(-rf_search.best_score_)
    xgb_best_rmse = np.sqrt(-xgb_search.best_score_)
    svr_best_rmse = np.sqrt(-svr_search.best_score_)
    
    print("RF best params:", rf_search.best_params_, "CV RMSE:", rf_best_rmse)
    print("XGB best params:", xgb_search.best_params_, "CV RMSE:", xgb_best_rmse)
    print("SVR best params:", svr_search.best_params_, "CV RMSE:", svr_best_rmse)
    
    return rf_search.best_estimator_, xgb_search.best_estimator_, svr_search.best_estimator_