# tests/test_train.py

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline


def _make_fake_df(n_rows: int = 80) -> pd.DataFrame:
    rng = np.random.RandomState(0)
    df = pd.DataFrame(
        {
            "f1": rng.normal(size=n_rows),
            "f2": rng.normal(size=n_rows),
            "target": rng.binomial(1, 0.4, size=n_rows),
        }
    )
    return df


@pytest.fixture()
def fake_paths(tmp_path: Path):
    artifacts_dir = tmp_path / "artifacts"
    processed_csv = tmp_path / "processed.csv"
    final_model_path = tmp_path / "model" / "final_model.joblib"
    return artifacts_dir, processed_csv, final_model_path


def test_evaluate_model_creates_plots_and_metrics(tmp_path: Path):
    # Import the module under test
    train_model = importlib.import_module("src.models.train_model")

    # Create simple pipeline that supports predict_proba
    X_test = pd.DataFrame({"f1": [0, 1, 2, 3], "f2": [1, 1, 0, 0]})
    y_test = np.array([0, 1, 0, 1])

    model = Pipeline(
        steps=[
            ("clf", DummyClassifier(strategy="prior")),
        ]
    ).fit(X_test, y_test)

    artifact_dir = tmp_path / "plots"
    metrics, cm_path, roc_path = train_model.evaluate_model(
        model, X_test, y_test, prefix="dummy", artifact_dir=artifact_dir
    )

    # Metric keys exist
    assert "dummy_accuracy" in metrics
    assert "dummy_precision" in metrics
    assert "dummy_recall" in metrics
    assert "dummy_f1" in metrics
    assert "dummy_roc_auc" in metrics

    # Files exist
    assert Path(cm_path).exists()
    assert Path(roc_path).exists()


def test_main_runs_end_to_end(monkeypatch, fake_paths):
    """
    Runs train_model.main() with:
    - a fake processed CSV
    - patched config constants (paths, random_state, test_size)
    - stubbed feature functions (split_features_target/build_preprocessor)
    - stubbed MLflow utils to no-op (so no tracking server needed)
    """
    artifacts_dir, processed_csv, final_model_path = fake_paths

    # Write fake CSV
    df = _make_fake_df(80)
    df.to_csv(processed_csv, index=False)

    # Import module
    train_model = importlib.import_module("src.models.train_model")

    # ---- Patch config values inside train_model module ----
    monkeypatch.setattr(train_model, "ARTIFACTS_DIR", artifacts_dir, raising=True)
    monkeypatch.setattr(train_model, "PROCESSED_DATA_PATH", processed_csv, raising=True)
    monkeypatch.setattr(train_model, "FINAL_MODEL_PATH", final_model_path, raising=True)
    monkeypatch.setattr(train_model, "RANDOM_STATE", 42, raising=True)
    monkeypatch.setattr(train_model, "TEST_SIZE", 0.25, raising=True)

    # ---- Stub out feature functions ----
    def split_features_target_stub(df_in: pd.DataFrame):
        X = df_in[["f1", "f2"]]
        y = df_in["target"].astype(int).to_numpy()
        return X, y

    # Preprocessor can just be "passthrough" for the test
    def build_preprocessor_stub():
        return "passthrough"

    monkeypatch.setattr(train_model, "split_features_target", split_features_target_stub, raising=True)
    monkeypatch.setattr(train_model, "build_preprocessor", build_preprocessor_stub, raising=True)

    # ---- Stub MLflow utils used by the script (no-op) ----
    class _DummyRun:
        def __enter__(self):  # context manager
            return self
        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(train_model, "start_run", lambda **kwargs: _DummyRun(), raising=True)
    monkeypatch.setattr(train_model, "enable_autolog", lambda *a, **k: None, raising=True)
    monkeypatch.setattr(train_model, "log_artifact", lambda *a, **k: None, raising=True)
    monkeypatch.setattr(train_model, "log_artifacts", lambda *a, **k: None, raising=True)
    monkeypatch.setattr(train_model, "log_dataset_info", lambda *a, **k: None, raising=True)
    monkeypatch.setattr(train_model, "log_json", lambda *a, **k: None, raising=True)
    monkeypatch.setattr(train_model, "log_metrics", lambda *a, **k: None, raising=True)
    monkeypatch.setattr(train_model, "log_params", lambda *a, **k: None, raising=True)
    monkeypatch.setattr(train_model, "log_text", lambda *a, **k: None, raising=True)

    # train_model also calls mlflow.set_tag directly; patch that too
    monkeypatch.setattr(train_model.mlflow, "set_tag", lambda *a, **k: None, raising=True)

    # Run
    train_model.main(random_state=42, test_size=0.25)

    # Assert outputs created locally
    assert (artifacts_dir / "metrics.json").exists()
    assert final_model_path.exists()

    # Plots should be created (evaluate_model saves them locally)
    plots_dir = artifacts_dir / "plots"
    assert plots_dir.exists()
    # At least some pngs should exist
    assert any(p.suffix == ".png" for p in plots_dir.glob("*.png"))