# Breast Cancer Classification — End-to-End ML Pipeline

A complete machine learning workflow on the Wisconsin Breast Cancer dataset
(built into scikit-learn): exploratory analysis, model comparison,
hyperparameter tuning, and model explainability with SHAP.

The goal isn't just a high accuracy number — it's demonstrating the full
lifecycle: understand the data, compare candidate models fairly, tune the
winner, and then *explain* what the model actually learned.

## Results

| Model | Test Accuracy |
|---|---|
| Random Forest | 96.5% |
| Logistic Regression | 95.6% |
| K-Nearest Neighbors | 95.6% |
| Support Vector Machine | 94.7% |

The Random Forest is then tuned with a 5-fold cross-validated grid search
over `n_estimators` and `max_depth`, and its predictions are explained with
SHAP values to identify which cell-nucleus features (e.g. worst radius,
worst concave points) drive the malignant/benign classification.

## Pipeline

```
main.py
 ├─ preprocess.py             load dataset, overview, class distribution
 ├─ evaluate.py               accuracy metrics, class-distribution pie chart
 ├─ visualize.py              correlation heatmap, feature importances,
 │                            tuning-curve plots
 ├─ train_model.py            train/test split, baseline Random Forest
 ├─ model_comparison.py       4 classifiers compared on identical splits
 ├─ hyperparameter_tuning.py  GridSearchCV for Random Forest + LogReg
 └─ explainability.py         SHAP TreeExplainer with a manual fallback plot
```

Running `main.py` executes the whole pipeline and pops up each plot in
sequence: class distribution → correlation heatmap → feature importances →
model comparison table → tuning curves → confusion matrix → SHAP summary.

## Setup

Requires Python 3.10+.

```bash
git clone https://github.com/YOUR_USERNAME/breast-cancer-classification.git
cd breast-cancer-classification
pip install -r requirements.txt
python main.py
```

No dataset download needed — the Wisconsin Breast Cancer dataset ships with
scikit-learn (569 samples, 30 numeric features).

## What I'd add next

- [ ] Save trained models with `joblib` instead of retraining every run
- [ ] Precision/recall trade-off analysis (in a medical context, false
      negatives matter more than raw accuracy)
- [ ] ROC curves and AUC for each model in the comparison
- [ ] A `--no-plots` flag for running headless

## License

MIT — see [LICENSE](LICENSE).
