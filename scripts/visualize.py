import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_feature_importance(model, feature_names, top_n=10):
    """Plots the top N most important features in the RandomForest model."""
    importances = model.feature_importances_
    indices = importances.argsort()[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=top_features, palette='viridis')
    plt.title(f'Top {top_n} Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df):
    """Displays a correlation heatmap of the features."""
    plt.figure(figsize=(14, 10))
    corr = df.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.show()


def plot_feature_importances(model, feature_names, top_n=10):
    """
    Plots the top N most important features from a trained model.

    Args:
        model: Trained classifier with `feature_importances_` attribute.
        feature_names: List of feature names (e.g., df.columns without target).
        top_n: How many top features to plot (default is 10).
    """
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })

    # Sort by importance descending
    top_features = importance_df.sort_values(by='Importance', ascending=False).head(top_n)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(top_features['Feature'][::-1], top_features['Importance'][::-1], color='royalblue')
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Most Important Features')
    plt.tight_layout()
    plt.show()


def plot_rf_tuning_results(cv_results):
    """
    Plots accuracy vs n_estimators for each max_depth in the Random Forest hyperparameter tuning.
    """
    # Convert results to DataFrame
    results_df = pd.DataFrame(cv_results)

    # Plot mean test accuracy for different parameter combinations
    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=results_df,
        x='param_n_estimators',
        y='mean_test_score',
        hue='param_max_depth',
        marker='o'
    )

    plt.title('Random Forest Tuning: Accuracy vs. n_estimators')
    plt.xlabel('Number of Trees (n_estimators)')
    plt.ylabel('Mean CV Accuracy')
    plt.legend(title='Max Depth')
    plt.tight_layout()
    plt.show()