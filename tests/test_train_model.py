import json

import pandas as pd
import pytest

from src.config import TARGET_COLUMN
from src.models.train_model import main as train_main


@pytest.fixture
def tiny_processed_df():
    """Return a tiny but valid processed dataset as a DataFrame."""
    return pd.DataFrame(
        {
            "age": [63, 45, 54, 60],
            "trestbps": [145, 130, 120, 140],
            "chol": [233, 250, 240, 220],
            "thalach": [150, 140, 130, 160],
            "oldpeak": [2.3, 1.4, 0.0, 1.5],
            "sex": [1, 0, 1, 1],
            "cp": [3, 2, 1, 0],
            "fbs": [1, 0, 0, 1],
            "restecg": [0, 1, 1, 0],
            "exang": [0, 1, 0, 1],
            "slope": [0, 1, 2, 1],
            "ca": [0, 0, 1, 2],
            "thal": [1, 2, 2, 3],
            TARGET_COLUMN: [1, 0, 0, 1],
        }
    )


def test_train_model_runs_and_saves_model(tmp_path, monkeypatch, tiny_processed_df):
    """
    End-to-end test:
    - Writes a tiny processed dataset to disk
    - Monkeypatches paths in train_model
    - Runs main()
    - Asserts that a model file and metrics.json are created
    """
    # Prepare temporary files/dirs
    processed_path = tmp_path / "heart_clean.csv"
    final_model_path = tmp_path / "heart_model.pkl"
    artifacts_dir = tmp_path / "artifacts"

    # Save synthetic processed data
    tiny_processed_df.to_csv(processed_path, index=False)

    # Monkeypatch config paths inside src.models.train_model
    monkeypatch.setattr("src.models.train_model.PROCESSED_DATA_PATH", processed_path)
    monkeypatch.setattr("src.models.train_model.FINAL_MODEL_PATH", final_model_path)
    monkeypatch.setattr("src.models.train_model.ARTIFACTS_DIR", artifacts_dir)

    # Use small test_size so we have a couple of train samples; CV handling
    # in train_model should adapt n_splits accordingly.
    train_main(random_state=0, test_size=0.5)

    # Check model file created
    assert final_model_path.exists(), "Final model file was not created."

    # Check metrics.json exists
    metrics_path = artifacts_dir / "metrics.json"
    assert metrics_path.exists(), "metrics.json was not created."

    # Load metrics.json and do basic structural checks
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    # We expect entries for both models: 'logreg' and 'rf'
    assert "logreg" in metrics, "Metrics for 'logreg' not found in metrics.json."
    assert "rf" in metrics, "Metrics for 'rf' not found in metrics.json."

    # Each model should at least have a ROC-AUC metric
    assert "logreg_roc_auc" in metrics["logreg"]
    assert "rf_roc_auc" in metrics["rf"]

    # Optionally: check that confusion matrix / ROC curve images exist
    # (prefixes are 'logreg' and 'rf' as in the train_model code)
    logreg_cm = artifacts_dir / "logreg_confusion_matrix.png"
    logreg_roc = artifacts_dir / "logreg_roc_curve.png"
    rf_cm = artifacts_dir / "rf_confusion_matrix.png"
    rf_roc = artifacts_dir / "rf_roc_curve.png"

    assert logreg_cm.exists(), "LogReg confusion matrix PNG not created."
    assert logreg_roc.exists(), "LogReg ROC curve PNG not created."
    assert rf_cm.exists(), "RandomForest confusion matrix PNG not created."
    assert rf_roc.exists(), "RandomForest ROC curve PNG not created."


def test_train_model_uses_mlflow_utils(tmp_path, monkeypatch, tiny_processed_df):
    """
    Test that the MLflow utility functions are called at least once.
    We mock them and assert they were invoked.
    """
    processed_path = tmp_path / "heart_clean.csv"
    final_model_path = tmp_path / "heart_model.pkl"
    artifacts_dir = tmp_path / "artifacts"

    tiny_processed_df.to_csv(processed_path, index=False)

    # Monkeypatch paths in the module under test
    monkeypatch.setattr("src.models.train_model.PROCESSED_DATA_PATH", processed_path)
    monkeypatch.setattr("src.models.train_model.FINAL_MODEL_PATH", final_model_path)
    monkeypatch.setattr("src.models.train_model.ARTIFACTS_DIR", artifacts_dir)

    # --- Mock MLflow utilities ---
    start_run_calls = []
    log_params_calls = []
    log_metrics_calls = []
    log_artifact_calls = []

    class DummyRunCtx:
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            start_run_calls.append(self.name)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    def fake_start_run(run_name: str):
        return DummyRunCtx(run_name)

    def fake_log_params(params):
        log_params_calls.append(params)

    def fake_log_metrics(metrics):
        log_metrics_calls.append(metrics)

    def fake_log_artifact(path):
        log_artifact_calls.append(path)

    monkeypatch.setattr("src.models.train_model.start_run", fake_start_run)
    monkeypatch.setattr("src.models.train_model.log_params", fake_log_params)
    monkeypatch.setattr("src.models.train_model.log_metrics", fake_log_metrics)
    monkeypatch.setattr("src.models.train_model.log_artifact", fake_log_artifact)

    # Run training
    train_main(random_state=0, test_size=0.5)

    # Assert MLflow-like utilities were called
    # We expect at least one run for each model: 'logreg' and 'rf'
    assert any("logreg" == name for name in start_run_calls), (
        "start_run not called for 'logreg'."
    )
    assert any("rf" == name for name in start_run_calls), (
        "start_run not called for 'rf'."
    )

    assert len(log_params_calls) >= 2, (
        "log_params should be called at least once per model."
    )
    assert len(log_metrics_calls) >= 2, (
        "log_metrics should be called at least once per model."
    )
    assert len(log_artifact_calls) >= 4, (
        "log_artifact should be called for multiple plots."
    )
