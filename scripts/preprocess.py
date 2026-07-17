import pandas as pd
import io
from sklearn.datasets import load_breast_cancer

def load_data():
    """Loads the breast cancer dataset and returns a DataFrame with features and target."""
    print(" Loading breast cancer dataset...\n")
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df, data

def describe_data(df):
    """Prints the dataset overview and descriptive statistics."""
    print("Dataset Overview:")
    buffer = io.StringIO()
    df.info(buf=buffer)
    print(buffer.getvalue())

    print(" Descriptive Statistics:")
    print(df.describe(include='all'))

def get_class_distribution(df):
    """Returns the distribution of the target classes."""
    return df['target'].value_counts()
