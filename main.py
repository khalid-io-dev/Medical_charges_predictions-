import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_preprocessing import load_and_clean_data, feature_engineering, prepare_data, prepare_robust_scaled_data
from src.visualizations import plot_histograms, plot_boxplots, plot_correlation_heatmap, plot_smoker_trends, plot_model_performance
from src.model_training import run_experiments
from src.hyperparameter_tuning import tune_models
from src.utils import evaluate_tuned_models
from src.predict import predict_charges

def main():
    # Configure plotting
    plt.style.use('default')
    sns.set_palette("husl")

    # Load and preprocess data
    df = load_and_clean_data('data/assurance-maladie.csv')
    
    Basic EDA
    print("Value counts for categorical columns:")
    print(df['sex'].value_counts())
    print("\n")
    print(df['smoker'].value_counts())
    print("\n")
    print(df['region'].value_counts())
    
    # Visualizations
    plot_histograms(df)
    plot_boxplots(df)
    plot_smoker_trends(df)
    
    # Feature engineering
    df_encoded = feature_engineering(df)
    plot_correlation_heatmap(df_encoded, "Correlation Heatmap with Engineered Features")
    
    # Prepare data
    (X_train_scaled, X_test_scaled, X_train_unscaled, X_test_unscaled,
     y_train, y_test, feature_names, scaler) = prepare_data(df_encoded)
    
    # Run experiments
    results_df = run_experiments(X_train_unscaled, X_test_unscaled, y_train, y_test,
                                X_train_scaled, X_test_scaled)
    
    # Save best experiment model
    best_exp_row = results_df.loc[results_df['rmse'].idxmin()]
    best_exp_model = best_exp_row['estimator']
    joblib.dump(best_exp_model, "models/best_experiment_model.joblib")
    print(f"Saved best experiment model: {best_exp_row['model']} ({best_exp_row['experiment']})")
    
    # Tune models
    best_rf, best_xgb, best_svr = tune_models(X_train_unscaled, X_train_scaled, y_train)
    
    # Evaluate tuned models
    tuned_results = evaluate_tuned_models([
        ("RandomForest_tuned", best_rf),
        ("XGBoost_tuned", best_xgb),
        ("SVR_tuned", best_svr)
    ], X_test_unscaled, X_test_scaled, y_test) 
    
    # Compare baseline and tuned
    baseline_df = results_df[results_df['experiment'] == 'baseline'][["model", "rmse", "mae", "r2"]]
    comparison = pd.concat([baseline_df, tuned_results], ignore_index=True)
    print("\nBaseline vs Tuned comparison:")
    print(comparison.sort_values("rmse").reset_index(drop=True))
    
    # Select best model dynamically
    best_row = comparison.loc[comparison['rmse'].idxmin()]
    best_model_name = best_row['model']
    if best_model_name == "RandomForest_tuned":
        best_model = best_rf
        X_test_final = X_test_unscaled
    elif best_model_name == "XGBoost_tuned":
        best_model = best_xgb
        X_test_final = X_test_unscaled
    else:
        best_model = best_svr
        X_test_final = X_test_scaled  
    print(f"\nSelected best model: {best_model_name} (RMSE={best_row['rmse']:.2f})")
    
    # Retrain best model on full training set
    if best_model_name == "SVR_tuned":
        best_model.fit(X_train_scaled, y_train)
    else:
        best_model.fit(X_train_unscaled, y_train)
    joblib.dump(best_model, "models/final_model.joblib")
    print("Final model saved to models/final_model.joblib")
    
    # Plot performance of best model
    y_pred = best_model.predict(X_test_final)
    plot_model_performance(y_test, y_pred, best_model_name)
    
    # CLI for predictions with error handling
    print("\nEnter details for prediction:")
    raw = {}
    try:
        raw['age'] = int(input("Age (integer, e.g., 45): "))
        if raw['age'] < 0:
            raise ValueError("Age must be non-negative.")
        raw['sex'] = input("Sex (male/female): ").lower()
        if raw['sex'] not in ['male', 'female']:
            raise ValueError("Sex must be 'male' or 'female'.")
        raw['bmi'] = float(input("BMI (float, e.g., 28.3): "))
        if raw['bmi'] <= 0:
            raise ValueError("BMI must be positive.")
        raw['children'] = int(input("Children (integer, e.g., 2): "))
        if raw['children'] < 0:
            raise ValueError("Children must be non-negative.")
        raw['smoker'] = input("Smoker (yes/no): ").lower()
        if raw['smoker'] not in ['yes', 'no']:
            raise ValueError("Smoker must be 'yes' or 'no'.")
        raw['region'] = input("Region (northeast/northwest/southeast/southwest): ").lower()
        if raw['region'] not in ['northeast', 'northwest', 'southeast', 'southwest']:
            raise ValueError("Region must be one of: northeast, northwest, southeast, southwest.")
        
        pred = predict_charges(raw)
        print(f"Estimated charges: ${pred:,.2f}")
    except ValueError as e:
        print(f"Error: {e}. Please try again with valid inputs.")

if __name__ == "__main__":
    main()