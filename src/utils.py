import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR

def evaluate_tuned_models(models, X_test_unscaled, X_test_scaled, y_test):
    """Evaluate tuned models on test set."""
    results = []
    for name, model in models:
        # Use scaled data for SVR, unscaled for others
        X_test_used = X_test_scaled if isinstance(model, SVR) else X_test_unscaled
        y_pred = model.predict(X_test_used)
        mse = mean_squared_error(y_test, y_pred)
        rmse = sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results.append({"model": name, "rmse": rmse, "mae": mae, "r2": r2})
        print(f"{name} -> RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.3f}")
    return pd.DataFrame(results)[["model", "rmse", "mae", "r2"]]