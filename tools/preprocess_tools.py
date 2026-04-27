"""
Tool for preprocessing tabular data: imputation, encoding, splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from langchain.tools import tool
import os

@tool
def run_preprocessing(
    raw_path: str,
    target_column: str,
    imputation_strategy: str = "mean",
    test_size: float = 0.2
) -> dict:
    """
    Load raw CSV, impute missing values, encode categorical columns,
    split into train/test, and save as .npy files.
    
    Args:
        raw_path: Path to raw CSV file.
        target_column: Name of the target/label column.
        imputation_strategy: "mean", "median", "most_frequent", or "drop".
        test_size: Fraction for test set.
    
    Returns:
        dict with keys:
            - "processed_paths": dict with "X_train", "X_test", "y_train", "y_test" paths
            - "input_shape": (n_features,)
            - "num_classes": number of unique labels
    """
    df = pd.read_csv(raw_path)
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Handle missing values in X
    if imputation_strategy == "drop":
        # Drop rows with any missing in X
        valid_idx = X.notna().all(axis=1)
        X = X[valid_idx]
        y = y[valid_idx]
    else:
        # Impute numeric columns with mean/median, categorical with most_frequent
        numeric_cols = X.select_dtypes(include=np.number).columns
        categorical_cols = X.select_dtypes(include='object').columns
        
        if imputation_strategy in ["mean", "median"]:
            for col in numeric_cols:
                if imputation_strategy == "mean":
                    X[col].fillna(X[col].mean(), inplace=True)
                else:
                    X[col].fillna(X[col].median(), inplace=True)
        # For categorical, always use most_frequent
        for col in categorical_cols:
            X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "missing", inplace=True)
    
    # Encode categorical features using one-hot encoding
    X = pd.get_dummies(X, drop_first=True)
    
    # Encode target if it's categorical
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
        num_classes = len(le.classes_)
    else:
        num_classes = len(np.unique(y))
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Save as .npy files
    os.makedirs("data/processed", exist_ok=True)
    X_train_path = "data/processed/X_train.npy"
    X_test_path = "data/processed/X_test.npy"
    y_train_path = "data/processed/y_train.npy"
    y_test_path = "data/processed/y_test.npy"
    
    np.save(X_train_path, X_train.values)
    np.save(X_test_path, X_test.values)
    np.save(y_train_path, y_train)
    np.save(y_test_path, y_test)
    
    return {
        "processed_paths": {
            "X_train": X_train_path,
            "X_test": X_test_path,
            "y_train": y_train_path,
            "y_test": y_test_path
        },
        "input_shape": X_train.shape[1],
        "num_classes": num_classes
    }