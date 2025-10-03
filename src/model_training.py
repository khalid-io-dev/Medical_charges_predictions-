import numpy as np
import pandas as pd
from math import sqrt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.data_preprocessing import prepare_robust_scaled_data
from xgboost import XGBRegressor

def evaluate_model_on_test(model, X_test, y_test, log_target=False):
    """Evaluate model and return RMSE, MAE, R2."""
    y_pred = model.predict(X_test)
    if log_target:
        y_pred = np.exp(y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return rmse, mae, r2

def train_and_evaluate(models_dict, Xtr, Xte, ytr, yte, scaled_for_linear=False, log_target=False):
    """Train and evaluate models."""
    results = []
    for name, model in models_dict.items():
        # Choose X based on model type (scaled for linear/SVR, unscaled for trees)
        if scaled_for_linear and name in ["LinearRegression", "SVR"]:
            Xtr_used = Xtr["scaled"]
            Xte_used = Xte["scaled"]
        else:
            Xtr_used = Xtr["unscaled"]
            Xte_used = Xte["unscaled"]
        
        if log_target:
            model.fit(Xtr_used, np.log(ytr))
            rmse, mae, r2 = evaluate_model_on_test(model, Xte_used, yte, log_target=True)
        else:
            model.fit(Xtr_used, ytr)
            rmse, mae, r2 = evaluate_model_on_test(model, Xte_used, yte, log_target=False)
        
        results.append({"model": name, "rmse": rmse, "mae": mae, "r2": r2, "estimator": model})
        print(f"{name} -> RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.3f}")
    return results

def run_experiments(X_train_unscaled, X_test_unscaled, y_train, y_test, X_train_scaled, X_test_scaled):
    """Run baseline and alternative scaling experiments."""
    models_base = {
        "LinearRegression": LinearRegression(),
        "SVR": SVR(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42, eval_metric='rmse')
    }
    
    # RobustScaler data
    X_train_rob_scaled, X_test_rob_scaled, rob_scaler = prepare_robust_scaled_data(
        X_train_unscaled, X_test_unscaled, y_train, y_test
    )
    
    experiments = {
        "baseline": {
            "log_target": False,
            "scaled_for_linear": True,
            "Xtr": {"unscaled": X_train_unscaled, "scaled": X_train_scaled},
            "Xte": {"unscaled": X_test_unscaled, "scaled": X_test_scaled}
        },
        "log_std": {
            "log_target": True,
            "scaled_for_linear": True,
            "Xtr": {"unscaled": X_train_unscaled, "scaled": X_train_scaled},
            "Xte": {"unscaled": X_test_unscaled, "scaled": X_test_scaled}
        },
        "log_robust": {
            "log_target": True,
            "scaled_for_linear": True,
            "Xtr": {"unscaled": X_train_unscaled, "scaled": X_train_rob_scaled},
            "Xte": {"unscaled": X_test_unscaled, "scaled": X_test_rob_scaled}
        },
        "robust_only": {
            "log_target": False,
            "scaled_for_linear": True,
            "Xtr": {"unscaled": X_train_unscaled, "scaled": X_train_rob_scaled},
            "Xte": {"unscaled": X_test_unscaled, "scaled": X_test_rob_scaled}
        }
    }
    
    all_results = []
    for exp_name, cfg in experiments.items():
        print(f"\nExperiment: {exp_name}")
        res = train_and_evaluate(models_base, cfg["Xtr"], cfg["Xte"], y_train, y_test,
                                scaled_for_linear=cfg["scaled_for_linear"], log_target=cfg["log_target"])
        for r in res:
            r['experiment'] = exp_name
        all_results.extend(res)
    
    results_df = pd.DataFrame(all_results)[["experiment", "model", "rmse", "mae", "r2", "estimator"]]
    print("\nSummary of experiments (sorted by RMSE):")
    print(results_df[["experiment", "model", "rmse", "mae", "r2"]].sort_values("rmse").reset_index(drop=True))
    
    return results_df