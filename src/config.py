import os
from pathlib import Path

# Base directory (repo root assumed two levels up from this file)
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "heart.csv"
PROCESSED_DATA_PATH = DATA_DIR / "heart_clean.csv"

MODELS_DIR = BASE_DIR / "models"
FINAL_MODEL_PATH = MODELS_DIR / "final_model.pkl"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
MLFLOW_DIR = BASE_DIR / "mlflow"

# MLflow
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DIR / 'mlflow.db'}"
MLFLOW_EXPERIMENT_NAME = "heart-disease-classification"

# Target column name
TARGET_COLUMN = "target"

# Example columns (adjust according to your CSV!)
NUMERIC_FEATURES = [
    "age", "trestbps", "chol", "thalach", "oldpeak"
]

CATEGORICAL_FEATURES = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
]

# Create directories if not exist
for p in [
    DATA_DIR,
    MODELS_DIR,
    ARTIFACTS_DIR,
    MLFLOW_DIR,
]:
    p.mkdir(parents=True, exist_ok=True)