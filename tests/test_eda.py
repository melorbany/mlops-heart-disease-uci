import importlib
from pathlib import Path

import pandas as pd
import pytest


# IMPORTANT: change this to the actual module path of the file you posted.
# Example: if the file is src/eda.py, keep "src.eda".
eda = importlib.import_module("src.eda.visualize")


@pytest.fixture
def sample_df():
    """
    Minimal dataframe that includes:
    - some numeric features (as configured in NUMERIC_FEATURES)
    - the TARGET_COLUMN
    """
    # Make sure we include at least a few numeric columns that are likely in NUMERIC_FEATURES.
    # If your NUMERIC_FEATURES differs, the tests still pass because the code filters by df.columns.
    data = {
        "age": [63, 67, 67, 37, 41, 56, 62, 57],
        "trestbps": [145, 160, 120, 130, 130, 120, 140, 120],
        "chol": [233, 286, 229, 250, 204, 236, 268, 354],
        # target column may be something else in your config; we will map it below in tests
        "target": [1, 1, 1, 0, 0, 1, 0, 0],
    }
    return pd.DataFrame(data)


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    # mimic artifacts base dir, and let create_eda_directory create artifacts/eda
    return tmp_path


def test_create_eda_directory_creates_dir(output_dir: Path):
    eda_dir = eda.create_eda_directory(output_dir)
    assert eda_dir.exists()
    assert eda_dir.is_dir()
    assert eda_dir.name == "eda"


def test_plot_histograms_saves_png(sample_df: pd.DataFrame, tmp_path: Path, monkeypatch):
    eda_dir = eda.create_eda_directory(tmp_path)

    # Ensure NUMERIC_FEATURES contains at least one of our columns to exercise plotting.
    monkeypatch.setattr(eda, "NUMERIC_FEATURES", ["age", "trestbps", "chol"], raising=True)

    out = eda.plot_histograms(sample_df, eda_dir)
    assert out == eda_dir / "histograms.png"
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_correlation_heatmap_saves_png(sample_df: pd.DataFrame, tmp_path: Path, monkeypatch):
    eda_dir = eda.create_eda_directory(tmp_path)

    monkeypatch.setattr(eda, "NUMERIC_FEATURES", ["age", "trestbps", "chol"], raising=True)
    monkeypatch.setattr(eda, "TARGET_COLUMN", "target", raising=True)

    out = eda.plot_correlation_heatmap(sample_df, eda_dir)
    assert out == eda_dir / "correlation_heatmap.png"
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_class_balance_raises_if_target_missing(sample_df: pd.DataFrame, tmp_path: Path, monkeypatch):
    eda_dir = eda.create_eda_directory(tmp_path)

    monkeypatch.setattr(eda, "TARGET_COLUMN", "target", raising=True)

    df_no_target = sample_df.drop(columns=["target"])
    with pytest.raises(ValueError, match="Target column 'target' not found"):
        eda.plot_class_balance(df_no_target, eda_dir)


def test_plot_class_balance_saves_png(sample_df: pd.DataFrame, tmp_path: Path, monkeypatch):
    eda_dir = eda.create_eda_directory(tmp_path)

    monkeypatch.setattr(eda, "TARGET_COLUMN", "target", raising=True)

    out = eda.plot_class_balance(sample_df, eda_dir)
    assert out == eda_dir / "class_balance.png"
    assert out.exists()
    assert out.stat().st_size > 0


def test_generate_eda_summary_writes_expected_text(sample_df: pd.DataFrame, tmp_path: Path, monkeypatch):
    eda_dir = eda.create_eda_directory(tmp_path)

    monkeypatch.setattr(eda, "NUMERIC_FEATURES", ["age", "trestbps", "chol"], raising=True)
    monkeypatch.setattr(eda, "TARGET_COLUMN", "target", raising=True)

    out = eda.generate_eda_summary(sample_df, eda_dir)
    assert out == eda_dir / "eda_summary.txt"
    assert out.exists()
    text = out.read_text(encoding="utf-8")

    assert "HEART DISEASE DATASET - EDA SUMMARY" in text
    assert "Dataset shape:" in text
    assert "Target distribution:" in text
    assert "Missing values per column:" in text
    assert "Numeric features summary:" in text
    # spot-check numeric features are mentioned
    assert "age:" in text
    assert "chol:" in text


def test_main_end_to_end_writes_all_outputs(sample_df: pd.DataFrame, tmp_path: Path, monkeypatch):
    """
    End-to-end test for main() using a temp CSV and temp output folder.
    """
    # Patch config constants used by the module
    monkeypatch.setattr(eda, "NUMERIC_FEATURES", ["age", "trestbps", "chol"], raising=True)
    monkeypatch.setattr(eda, "TARGET_COLUMN", "target", raising=True)

    data_path = tmp_path / "processed.csv"
    sample_df.to_csv(data_path, index=False)

    eda.main(data_path=data_path, output_base_dir=tmp_path)

    eda_dir = tmp_path / "eda"
    assert (eda_dir / "histograms.png").exists()
    assert (eda_dir / "correlation_heatmap.png").exists()
    assert (eda_dir / "class_balance.png").exists()
    assert (eda_dir / "eda_summary.txt").exists()