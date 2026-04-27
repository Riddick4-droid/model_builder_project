"""
Tools that agents can use to perform real actions.
Each tool is decorated with @tool from langchain.
"""

import mlflow
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from langchain.tools import tool

@tool
def train_sklearn_model(X_path: str, y_path: str, params: dict) -> dict:
    """
    Train a RandomForest classifier using provided feature and label files.
    
    Args:
        X_path: Path to features (CSV or .npy file)
        y_path: Path to labels (CSV or .npy file)
        params: Dictionary of hyperparameters for RandomForestClassifier,
                e.g., {"n_estimators": 100, "max_depth": 5, "random_state": 42}
    
    Returns:
        dict with keys:
            - "accuracy": training accuracy (float)
            - "run_id": MLflow run ID (string)
    
    Raises:
        ValueError: if file format not supported or loading fails.
    """
    # Load features
    if X_path.endswith('.csv'):
        X = pd.read_csv(X_path).values
    elif X_path.endswith('.npy'):
        X = np.load(X_path)
    else:
        raise ValueError(f"Unsupported feature file format: {X_path}")
    
    # Load labels
    if y_path.endswith('.csv'):
        y = pd.read_csv(y_path).values.ravel()
    elif y_path.endswith('.npy'):
        y = np.load(y_path)
    else:
        raise ValueError(f"Unsupported label file format: {y_path}")
    
    # Ensure parameters are valid for RandomForest
    default_params = {
        'n_estimators': 100,
        'random_state': 42
    }
    default_params.update(params)
    
    model = RandomForestClassifier(**default_params)
    model.fit(X, y)
    
    # Evaluate on training set (for demo; use validation in real system)
    preds = model.predict(X)
    accuracy = accuracy_score(y, preds)
    
    # Log with MLflow
    with mlflow.start_run() as run:
        mlflow.log_params(default_params)
        mlflow.log_metric("train_accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")
        run_id = run.info.run_id
    
    return {
        "accuracy": round(accuracy, 4),
        "run_id": run_id
    }