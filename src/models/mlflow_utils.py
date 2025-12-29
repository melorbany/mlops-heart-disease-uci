from typing import Dict

import mlflow
from src.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI


def init_mlflow() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def start_run(run_name: str = None):
    init_mlflow()
    return mlflow.start_run(run_name=run_name)


def log_params(params: Dict):
    mlflow.log_params(params)


def log_metrics(metrics: Dict):
    mlflow.log_metrics(metrics)


def log_artifact(path: str):
    mlflow.log_artifact(path)
