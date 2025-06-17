
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix, 
    mean_absolute_error, 
    mean_squared_error, 
    r2_score
)

from xgboost import XGBClassifier, XGBRegressor

from sklearn.preprocessing import LabelEncoder

from common_process import preprocess_data

def load_data(path):
    """Load CSV data into a DataFrame."""
    return pd.read_csv(path)


def split_features_labels(data, label):
    """Split into X and y, and label-encode y if it's non-numeric."""
    X = data.copy()
    y = X.pop(label)

    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    return X, y

def scale_features(X):
    """One-hot encode categoricals and scale numerical features."""
    # Separate into numerical and categorical
    num_vals = X.select_dtypes(include=['number'])
    cat_vals = X.select_dtypes(exclude=['number'])

    if not cat_vals.empty:
        cat_vals = pd.get_dummies(cat_vals, drop_first=True)

    scaler = StandardScaler()
    num_vals_scaled = scaler.fit_transform(num_vals)
    num_vals = pd.DataFrame(num_vals_scaled, columns=num_vals.columns, index=X.index)

    if not cat_vals.empty:
        X_transformed = pd.concat([num_vals, cat_vals], axis=1)
    else:
        X_transformed = num_vals

    return X_transformed


def split_data(X, y, test_size=0.2, random_state=42):
    """Split into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_and_evaluate(models, X_train, y_train, X_test, y_test):
    """Train multiple models and return their scores."""
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {
            "accuracy": accuracy,
            "report": classification_report(y_test, y_pred),
            "conf_matrix": confusion_matrix(y_test, y_pred),
        }
    return results


def pipeline(data, label='productive_day', type='classification', threshold=None):
    """General pipeline to streamline process from CSV or DataFrame to evaluation."""
    if isinstance(data, str):
        df = load_data(data)  # CSV path
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError("Expected a CSV file path or a pandas DataFrame.")
    
    processed = preprocess_data(df)
    X, y = split_features_labels(processed, label)
    
    if type == 'classification':
        unique_classes = y.unique()
        if len(unique_classes) != 2:
            raise ValueError(f"Expected 2 classes for binary, but got {len(unique_classes)}.")
        
        # Sort to make the mapping deterministic
        unique_classes_sorted = sorted(unique_classes)
        mapping = {unique_classes_sorted[0]: 0, unique_classes_sorted[1]: 1}
        
        y = y.map(mapping)
        if y.isnull().any():
            raise ValueError("Some values were not mapped. Check your data.")
    else:
        y = y

    X_scaled = scale_features(X)

    X_train, X_test, y_train, y_test = split_data(X_scaled, y)

    if type == 'classification':
        models = {
            "Logistic Regression": LogisticRegression(), 
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "XGBoost": XGBClassifier(eval_metric='logloss')
        }
    else:
        models = {
            "Linear Regression": LinearRegression(), 
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "XGBoost": XGBRegressor()
        }
    

    results = {"type": type}

    max_accuracy = 0.0

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if type == 'classification':
            accuracy = accuracy_score(y_test, y_pred)
            max_accuracy = max(max_accuracy, accuracy)
            results[name] = {
                "accuracy": accuracy,
                "report": classification_report(y_test, y_pred),
                "conf_matrix": confusion_matrix(y_test, y_pred),
            }
        else:
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            max_accuracy = max(max_accuracy, r2)
            results[name] = {
                "MAE": mae,
                "MSE": mse,
                "R2": r2,
            }
    
    return results, processed, max_accuracy





