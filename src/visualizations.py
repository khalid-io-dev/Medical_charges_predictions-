import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_histograms(df, figsize=(10, 6)):
    """Plot histograms of numerical columns."""
    df.hist(figsize=figsize)
    plt.show()

def plot_boxplots(df, columns=['bmi', 'charges']):
    """Plot boxplots for specified columns."""
    for col in columns:
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.show()

def plot_correlation_heatmap(df, title="Correlation Heatmap"):
    """Plot correlation heatmap."""
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(title)
    plt.show()

def plot_smoker_trends(df):
    """Plot cumulative smokers by age."""
    smokers = df[df["smoker"] == "yes"]
    smoker_count_by_age = smokers.groupby("age").size().reset_index(name="count")
    smoker_count_by_age["cumulative_smokers"] = smoker_count_by_age["count"].cumsum()
    
    plt.figure(figsize=(10, 6))
    plt.plot(smoker_count_by_age["age"], smoker_count_by_age["cumulative_smokers"], marker="o")
    plt.xlabel("Age")
    plt.ylabel("Cumulative Number of Smokers")
    plt.title("Age vs. Cumulative Smokers in Dataset")
    plt.grid(True)
    plt.show()

def plot_model_performance(y_test, y_pred, model_name):
    """Plot predicted vs actual and residuals."""
    # Predicted vs Actual
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    minv = min(y_test.min(), y_pred.min())
    maxv = max(y_test.max(), y_pred.max())
    plt.plot([minv, maxv], [minv, maxv], color="red", linewidth=1)
    plt.xlabel("Actual Charges")
    plt.ylabel("Predicted Charges")
    plt.title(f"Predicted vs Actual - {model_name}")
    plt.grid(True)
    plt.show()
    
    # Residuals distribution
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 4))
    sns.histplot(residuals, kde=True)
    plt.title(f"Residuals Distribution - {model_name}")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.show()
    
    # Residuals vs Predicted
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, color='red', linewidth=1)
    plt.xlabel("Predicted Charges")
    plt.ylabel("Residuals")
    plt.title(f"Residuals vs Predicted - {model_name}")
    plt.show()