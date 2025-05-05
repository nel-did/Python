import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    # Example preprocessing steps
    df = df.dropna()  # Drop missing values
    X = df.drop('target_column', axis=1)  # Replace 'target_column' with the actual target column name
    y = df['target_column']  # Replace 'target_column' with the actual target column name
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(report)

def save_model(model, filename):
    joblib.dump(model, filename)

if __name__ == "__main__":
    df = load_data(r"c:\Users\Nelson's lappy\OneDrive\Desktop\student_habits_performance.csv")
    X, y = preprocess_data(df)
    model, X_test, y_test = train_model(X, y)
    evaluate_model(model, X_test, y_test)
    save_model(model, 'student_performance_model.pkl')