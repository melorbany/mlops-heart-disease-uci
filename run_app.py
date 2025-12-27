# run_app.py
import shutil
import subprocess
import sys
from pathlib import Path

# Import paths from your config module
from src.config import BASE_DIR, DATA_DIR, ARTIFACTS_DIR, MLFLOW_DIR, MODELS_DIR, MLRUN_DIR



def clean_directories() -> None:
    """Delete important output directories (if they exist) before running the pipeline."""
    print("=== Cleaning previous run directories ===")

    # Directories to clean. You can add/remove as needed.
    clean_dirs = [
        DATA_DIR,
        ARTIFACTS_DIR,
        MLFLOW_DIR,
        MODELS_DIR,
        MLRUN_DIR,
    ]

    for path in clean_dirs:
        if path.exists():
            print(f" - Removing: {path}")
            shutil.rmtree(path, ignore_errors=True)
        else:
            print(f" - Skipping (not found): {path}")


def run(cmd: str) -> None:
    """Run a shell command, printing it first and exiting on failure."""
    print(f"\n>>> Running command: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True)
    except KeyboardInterrupt:
        # User pressed Ctrl+C while this command was running
        print("\n!!! Execution interrupted by user. Exiting.")
        sys.exit(130)  # 130 is the conventional exit code for SIGINT
    if result.returncode != 0:
        print(f"!!! Command failed with exit code {result.returncode}: {cmd}")
        sys.exit(result.returncode)



def main() -> None:
    print("=== MLOps Heart Disease pipeline + API launcher ===")

    # 0. Clean previous outputs
    print("\n[Step 0/5] Cleaning previous data...")
    clean_directories()

    # 1. Download data
    print("\n[Step 1/5] Downloading data...")
    run("python -m src.data.download_data")

    # 2. Convert UCI data to CSV
    print("\n[Step 2/5] Converting UCI data to CSV...")
    run("python -m src.data.convert_uci_to_csv")

    # 3. Preprocess CSV files
    print("\n[Step 3/5] Converting UCI data to CSV...")
    run("python -m src.data.preprocess")

    # 4. Train model
    print("\n[Step 4/5] Training model...")
    run("python -m src.models.train_model")

    # 5. Start FastAPI app with uvicorn
    print("\n[Step 5/5] Starting API server with uvicorn...")
    run("uvicorn src.api.main:app --reload")

    # Execution remains here while uvicorn is running


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Catch any stray Ctrl+C not caught inside run()
        print("\nScript interrupted by user.")
        sys.exit(130)