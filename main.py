from sklearn.metrics import accuracy_score
from scripts.explainability import explain_model_with_shap
from scripts.visualize import plot_rf_tuning_results
from sklearn.metrics import classification_report, confusion_matrix
from scripts.visualize import plot_feature_importances
import seaborn as sns
import matplotlib.pyplot as plt
from scripts.hyperparameter_tuning import tune_random_forest, tune_logistic_regression
from scripts.model_comparison import compare_models
from scripts.preprocess import load_data, describe_data, get_class_distribution
from scripts.train_model import split_data, train_rf_model
from scripts.evaluate import evaluate_model, plot_class_distribution
from scripts.visualize import plot_feature_importance, plot_correlation_heatmap


def main():
    df, data = load_data()
    describe_data(df)

    target_counts = get_class_distribution(df)
    print("\n Visualizing Class Distribution...")
    plot_class_distribution(target_counts)

    print("\n Visualizing Correlation Heatmap...")
    plot_correlation_heatmap(df.drop('target', axis=1))

    print("\n Preparing data for training...")
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")

    model = train_rf_model(X_train, y_train)
    y_pred, acc = evaluate_model(model, X_test, y_test)

    print("\n Visualizing Feature Importances...")
    plot_feature_importance(model, df.drop('target', axis=1).columns)

    print("\n Comparing multiple models...")
    model_comparison_df = compare_models(X_train, X_test, y_train, y_test)
    print("\n Model Comparison Results:")
    print(model_comparison_df.to_string(index=False))

    print("\n Tuning Random Forest...")
    best_rf, rf_params, rf_score, rf_cv_results = tune_random_forest(X_train, y_train)
    print(f" Best Random Forest Params: {rf_params}")
    print(f" Best Cross-Validated Accuracy: {rf_score * 100:.2f}%")

    print("\n🔧 Tuning Logistic Regression...")
    best_lr, lr_params, lr_score = tune_logistic_regression(X_train, y_train)
    print(f" Best Logistic Regression Params: {lr_params}")
    print(f" Best Cross-Validated Accuracy: {lr_score * 100:.2f}%")

    y_pred_tuned = best_rf.predict(X_test)
    print(f"\n Tuned Random Forest Test Accuracy: {accuracy_score(y_test, y_pred_tuned) * 100:.2f}%")

    #Confusion Matrix
    print("\n Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=['Malignant', 'Benign'],
                yticklabels=['Malignant', 'Benign'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

    #Classification Report
    print("\n Classification Report:")
    report = classification_report(y_test, y_pred, target_names=['Malignant', 'Benign'])
    print(report)

    #Plot top 10 feature importances
    X = df.drop('target', axis=1)
    plot_feature_importances(model, X.columns, top_n=10)

    #Visualize tuning results from the earlier grid search
    plot_rf_tuning_results(rf_cv_results)

    print("\n Explaining the Tuned Random Forest Model with SHAP...")
    explain_model_with_shap(best_rf, X_train, X_test)


if __name__ == "__main__":
    main()
