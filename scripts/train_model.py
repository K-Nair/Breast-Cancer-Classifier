from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def split_data(df):
    """Splits the data into train and test sets."""
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

def train_rf_model(X_train, y_train):
    """Trains a RandomForestClassifier and returns the trained model."""
    print("\n Training RandomForestClassifier...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    print("Model training completed!")
    return model
