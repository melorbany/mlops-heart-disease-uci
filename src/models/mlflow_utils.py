from __future__ import annotations

import getpass
import json
import platform
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Union

import mlflow

# If you have these constants already
from src.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI

JSONLike = Union[dict, list, str, int, float, bool, None]


def init_mlflow(
    tracking_uri: str = MLFLOW_TRACKING_URI,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def start_run(
    run_name: Optional[str] = None,
    tags: Optional[Mapping[str, str]] = None,
    nested: bool = False,
    description: Optional[str] = None,
):
    """
    Starts a run and logs useful default tags.
    """
    init_mlflow()

    run = mlflow.start_run(run_name=run_name, nested=nested)

    # Default tags to add context
    default_tags: Dict[str, str] = {
        "user": getpass.getuser(),
        "host": socket.gethostname(),
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    # Optional: add git info if available
    git_tags = _try_get_git_tags()
    default_tags.update(git_tags)

    # Merge user tags (user overrides defaults if same key)
    merged_tags = {**default_tags, **(dict(tags) if tags else {})}

    mlflow.set_tags(merged_tags)

    if description:
        # MLflow uses "mlflow.note.content" for run notes in many UIs
        mlflow.set_tag("mlflow.note.content", description)

    return run


def log_params(params: Mapping[str, Any], *, flatten: bool = True, sep: str = ".") -> None:
    """
    Logs parameters. Optionally flattens nested dicts to avoid losing structure.
    MLflow params are string-ish; we stringify non-scalars safely.
    """
    to_log = _flatten_dict(params, sep=sep) if flatten else dict(params)
    safe = {k: _to_param_value(v) for k, v in to_log.items()}
    mlflow.log_params(safe)


def log_metrics(
    metrics: Mapping[str, Union[int, float]],
    *,
    step: Optional[int] = None,
) -> None:
    """
    Logs metrics. If step is provided, logs each metric with that step.
    """
    if step is None:
        mlflow.log_metrics(dict(metrics))
    else:
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v), step=step)


def log_artifact(path: Union[str, Path], artifact_path: Optional[str] = None) -> None:
    mlflow.log_artifact(str(path), artifact_path=artifact_path)


def log_artifacts(dir_path: Union[str, Path], artifact_path: Optional[str] = None) -> None:
    mlflow.log_artifacts(str(dir_path), artifact_path=artifact_path)


def log_text(text: str, artifact_file: str) -> None:
    """
    Log a text blob as an artifact, e.g. artifact_file="reports/summary.txt".
    """
    mlflow.log_text(text, artifact_file)


def log_json(data: JSONLike, artifact_file: str, *, indent: int = 2) -> None:
    """
    Log JSON-serializable data as an artifact.
    """
    mlflow.log_text(json.dumps(data, indent=indent, ensure_ascii=False), artifact_file)


def log_dataset_info(
    *,
    name: str,
    version: Optional[str] = None,
    source: Optional[str] = None,
    n_rows: Optional[int] = None,
    n_cols: Optional[int] = None,
    hash: Optional[str] = None,
) -> None:
    """
    Convenience: log dataset metadata as tags/params.
    """
    tags: Dict[str, str] = {"dataset.name": name}
    if version:
        tags["dataset.version"] = version
    if source:
        tags["dataset.source"] = source
    if hash:
        tags["dataset.hash"] = hash
    mlflow.set_tags(tags)

    # numeric info can be metrics or params; I prefer params for sizes
    p: Dict[str, Any] = {}
    if n_rows is not None:
        p["dataset.n_rows"] = n_rows
    if n_cols is not None:
        p["dataset.n_cols"] = n_cols
    if p:
        log_params(p, flatten=False)


def enable_autolog(framework: str = "sklearn", **kwargs: Any) -> None:
    """
    Optional: one-liner to enable MLflow autologging.
    """
    if framework == "sklearn":
        mlflow.sklearn.autolog(**kwargs)
    elif framework == "xgboost":
        mlflow.xgboost.autolog(**kwargs)
    elif framework == "lightgbm":
        mlflow.lightgbm.autolog(**kwargs)
    elif framework == "pytorch":
        mlflow.pytorch.autolog(**kwargs)
    else:
        raise ValueError(f"Unsupported framework for autolog: {framework}")


# -------------------------
# Helpers
# -------------------------


def _to_param_value(v: Any) -> str:
    """
    MLflow params are stored as strings.
    """
    if v is None:
        return "null"
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    # For lists/dicts/objects, serialize to JSON-ish string
    try:
        return json.dumps(v, default=str, ensure_ascii=False)
    except Exception:
        return str(v)


def _flatten_dict(d: Mapping[str, Any], *, sep: str = ".", prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, Mapping):
            out.update(_flatten_dict(v, sep=sep, prefix=key))
        else:
            out[key] = v
    return out


def _try_get_git_tags() -> Dict[str, str]:
    """
    Best-effort git metadata without adding dependencies.
    """
    try:
        import subprocess

        def cmd(args: Iterable[str]) -> str:
            return subprocess.check_output(list(args), stderr=subprocess.DEVNULL).decode().strip()

        sha = cmd(["git", "rev-parse", "HEAD"])
        branch = cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        is_dirty = "true" if cmd(["git", "status", "--porcelain"]) else "false"
        return {"git.sha": sha, "git.branch": branch, "git.dirty": is_dirty}
    except Exception:
        return {}
