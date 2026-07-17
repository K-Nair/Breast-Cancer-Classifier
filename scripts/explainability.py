import shap
import matplotlib.pyplot as plt
import numpy as np


def explain_model_with_shap(model, X_train, X_test):
    print("Explaining the Tuned Random Forest Model with SHAP...")

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # If shap_values is a list (binary classification), pick class 1
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values_to_plot = shap_values[1]
        else:
            shap_values_to_plot = shap_values

        # Fix for 3D shap_values like (n_samples, n_features, 2)
        if len(shap_values_to_plot.shape) == 3:
            shap_values_to_plot = shap_values_to_plot[:, :, 0]

        # Use SHAP's newer Explanation format
        shap_exp = shap.Explanation(
            values=shap_values_to_plot,
            base_values=explainer.expected_value[1] if isinstance(explainer.expected_value,
                                                                  list) else explainer.expected_value,
            data=X_test,
            feature_names=X_test.columns
        )

        shap.plots.bar(shap_exp)

    except Exception as e:
        print(f" SHAP bar plot failed: {e}")
        print("Trying manual workaround with average absolute SHAP values...")

        try:
            #Fallback: manual barh plot
            mean_abs_shap = np.abs(shap_values_to_plot).mean(axis=0)
            sorted_idx = np.argsort(mean_abs_shap)[-10:]
            feature_names = np.array(X_test.columns)

            plt.figure(figsize=(10, 6))
            plt.barh(range(10), mean_abs_shap[sorted_idx], align='center')
            plt.yticks(range(10), feature_names[sorted_idx])
            plt.xlabel("Mean |SHAP value|")
            plt.title("Top 10 Important Features by SHAP (manual fallback)")
            plt.tight_layout()
            plt.show()

        except Exception as inner_e:
            print(f" Manual SHAP plot also failed: {inner_e}")
