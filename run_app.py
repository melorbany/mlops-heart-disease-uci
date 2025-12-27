# run_app.py
import os
import shutil
import subprocess
import sys

# Import paths from your config module
from src.config import (
    DATA_DIR,
    ARTIFACTS_DIR,
    MLFLOW_DIR,
    MODELS_DIR,
    MLRUN_DIR,
)


def is_running_in_docker() -> bool:
    """
    Heuristic to detect Docker.
    You can also rely solely on an explicit env var if you prefer.
    """
    if os.environ.get("RUN_IN_DOCKER") == "1":
        return True
    # Fallback heuristics: presence of /.dockerenv, etc.
    return os.path.exists("/.dockerenv")


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

    in_docker = is_running_in_docker()
    print(f"Running in Docker: {in_docker}")

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
    if in_docker:
        # Inside Docker: no reload, listen on all interfaces, port 8000
        uvicorn_cmd = "uvicorn src.api.main:app --host 0.0.0.0 --port 8000"
    else:
        # Local dev: keep reload for convenience
        uvicorn_cmd = "uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000"

    run(uvicorn_cmd)

    # Execution remains here while uvicorn is running


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Catch any stray Ctrl+C not caught inside run()
        print("\nScript interrupted by user.")
        sys.exit(130)
