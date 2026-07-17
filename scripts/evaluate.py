import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def evaluate_model(model, X_test, y_test):
    """Evaluates the model and prints accuracy and prediction stats."""
    print("\n Making predictions on test set...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    correct = sum(y_pred == y_test)

    print(f"\n Model Accuracy: {accuracy * 100:.2f}%")
    print(f"\n Results Summary:")
    print(f"   • Test accuracy: {accuracy * 100:.2f}%")
    print(f"   • Correct predictions: {correct}/{len(y_test)}")

    return y_pred, accuracy

def plot_class_distribution(target_counts):
    """Plots a pie chart of the class distribution with correct labels/colors."""
    labels = ['Benign', 'Malignant'] if target_counts.index[0] == 1 else ['Malignant', 'Benign']
    colors = ['green', 'red']

    plt.figure(figsize=(6, 6))
    plt.pie(target_counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Distribution of Breast Cancer Diagnosis')
    plt.axis('equal')
    plt.show()