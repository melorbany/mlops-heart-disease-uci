from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MLFLOW_DIR = BASE_DIR / "mlflow"

# Create directories if not exist
for p in [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    ARTIFACTS_DIR,
    MLFLOW_DIR,
]:
    p.mkdir(parents=True, exist_ok=True)

# MLflow
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DIR / 'mlflow.db'}"
MLFLOW_EXPERIMENT_NAME = "heart-disease-classification"


FINAL_MODEL_PATH = MODELS_DIR / "final_model.pkl"

TARGET_COLUMN = "target"

NUMERIC_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]

CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]

RANDOM_STATE = 42
TEST_SIZE = 0.2

MODEL_TYPE = "logreg"  # or "random_forest"

