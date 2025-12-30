import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

import mlflow
from src.config import (
    ARTIFACTS_DIR,
    FINAL_MODEL_PATH,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.features.build_features import build_preprocessor, split_features_target

# Updated imports: use richer mlflow_utils helpers
from src.models.mlflow_utils import (
    enable_autolog,
    log_artifact,
    log_artifacts,
    log_dataset_info,
    log_json,
    log_metrics,
    log_params,
    log_text,
    start_run,
)

matplotlib.use("Agg")  # non-interactive backend


def evaluate_model(
    model: Pipeline,
    X_test,
    y_test,
    *,
    prefix: str,
    artifact_dir: Path,
) -> Tuple[Dict[str, float], str, str]:
    y_pred = model.predict(X_test)

    # Handle models that may not expose predict_proba (not expected here, but safer)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        # Fall back to decision_function if needed
        y_scores = model.decision_function(X_test)
        # convert to 0-1 via min-max to keep roc_auc_score usable (not ideal but prevents crash)
        y_prob = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min() + 1e-12)

    metrics = {
        f"{prefix}_accuracy": float(accuracy_score(y_test, y_pred)),
        f"{prefix}_precision": float(precision_score(y_test, y_pred, zero_division=0)),
        f"{prefix}_recall": float(recall_score(y_test, y_pred, zero_division=0)),
        f"{prefix}_f1": float(f1_score(y_test, y_pred, zero_division=0)),
        f"{prefix}_roc_auc": float(roc_auc_score(y_test, y_prob)),
    }

    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Confusion Matrix
    fig_cm, ax_cm = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax_cm)
    ax_cm.set_title(f"{prefix} - Confusion Matrix")
    cm_path = artifact_dir / f"{prefix}_confusion_matrix.png"
    fig_cm.savefig(cm_path, bbox_inches="tight")
    plt.close(fig_cm)

    # ROC Curve
    fig_roc, ax_roc = plt.subplots()
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax_roc)
    ax_roc.set_title(f"{prefix} - ROC Curve")
    roc_path = artifact_dir / f"{prefix}_roc_curve.png"
    fig_roc.savefig(roc_path, bbox_inches="tight")
    plt.close(fig_roc)

    return metrics, str(cm_path), str(roc_path)


def main(random_state: int = RANDOM_STATE, test_size: float = TEST_SIZE) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Optional: enable autologging for sklearn (keeps your manual logging too)
    # If you prefer only manual logging, delete this line.
    enable_autolog("sklearn", log_models=False)  # we log the final model ourselves

    df = pd.read_csv(PROCESSED_DATA_PATH, na_values="?")
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    preprocessor = build_preprocessor()

    models = {
        "logreg": LogisticRegression(max_iter=1000, solver="lbfgs"),
        "rf": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    results: Dict[str, Dict[str, float]] = {}
    best_model_name: Optional[str] = None
    best_auc = -np.inf
    best_model: Optional[Pipeline] = None

    for name, clf in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("clf", clf),
            ]
        )

        run_tags = {
            "model.family": "sklearn",
            "model.name": name,
            "data.path": str(PROCESSED_DATA_PATH),
        }

        with start_run(
            run_name=name,
            tags=run_tags,
            description="Training + CV + evaluation with artifacts (CM/ROC) and dataset metadata.",
        ):
            # Log dataset context (visible in MLflow UI as tags/params)
            log_dataset_info(
                name=(
                    PROCESSED_DATA_PATH.stem
                    if hasattr(PROCESSED_DATA_PATH, "stem")
                    else "processed_data"
                ),
                source=str(PROCESSED_DATA_PATH),
                n_rows=int(df.shape[0]),
                n_cols=int(df.shape[1]),
            )

            # Log split sizes (handy for debugging)
            log_params(
                {
                    "random_state": random_state,
                    "test_size": test_size,
                    "n_rows_total": int(len(X)),
                    "n_rows_train": int(len(X_train)),
                    "n_rows_test": int(len(X_test)),
                },
                flatten=False,
            )

            # Log model hyperparams (namespaced)
            log_params({"clf": clf.get_params(), "pipeline": {"steps": ["preprocessor", "clf"]}})

            # ---- Robust cross-validation ----
            n_samples = len(X_train)
            n_splits = min(5, n_samples)  # cannot exceed n_samples

            cv_mean = None
            cv_std = None
            cv_scores = None

            if n_splits <= 2:
                # Not enough data for CV
                pipeline.fit(X_train, y_train)
                msg = (
                    "CV skipped due to insufficient training samples. "
                    f"n_train={n_samples}, n_splits={n_splits}"
                )
                log_text(msg, artifact_file=f"reports/{name}_cv_note.txt")
                mlflow.set_tag("cv.enabled", "false")
            else:
                mlflow.set_tag("cv.enabled", "true")
                mlflow.set_tag("cv.n_splits", str(n_splits))

                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
                cv_mean = float(np.mean(cv_scores))
                cv_std = float(np.std(cv_scores))

                # Log fold scores as an artifact for detail
                log_json(
                    {
                        "cv_scores_roc_auc": [float(x) for x in cv_scores],
                        "mean": cv_mean,
                        "std": cv_std,
                    },
                    artifact_file=f"reports/{name}_cv_scores.json",
                )

                pipeline.fit(X_train, y_train)

            # Evaluate
            metrics, cm_path, roc_path = evaluate_model(
                pipeline,
                X_test,
                y_test,
                prefix=name,
                artifact_dir=ARTIFACTS_DIR / "plots",
            )

            # Add CV metrics if available
            if cv_mean is not None:
                metrics[f"{name}_cv_roc_auc_mean"] = cv_mean
            if cv_std is not None:
                metrics[f"{name}_cv_roc_auc_std"] = cv_std

            # Log metrics and plots
            log_metrics(metrics)
            log_artifact(cm_path, artifact_path="plots")
            log_artifact(roc_path, artifact_path="plots")

            # Also log local artifacts directory contents (optional but useful)
            # Comment out if ARTIFACTS_DIR is large.
            log_artifacts(ARTIFACTS_DIR, artifact_path="local_artifacts_snapshot")

            results[name] = metrics

            if metrics[f"{name}_roc_auc"] > best_auc:
                best_auc = metrics[f"{name}_roc_auc"]
                best_model_name = name
                best_model = pipeline

    # Save metrics to JSON (local)
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    # Save best model (local)
    if best_model is None:
        raise RuntimeError("No model was trained; best_model is None.")

    FINAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, FINAL_MODEL_PATH)

    # Log final selection in its own run (optional but clean)
    with start_run(run_name="final_model", tags={"stage": "selection"}):
        log_params(
            {
                "best_model_name": best_model_name,
                "best_model_test_roc_auc": float(best_auc),
                "final_model_path": str(FINAL_MODEL_PATH),
            },
            flatten=False,
        )
        log_artifact(metrics_path, artifact_path="reports")
        log_artifact(FINAL_MODEL_PATH, artifact_path="model")

    print(f"Best model: {best_model_name} with ROC-AUC {best_auc:.4f}")
    print(f"Saved final model to {FINAL_MODEL_PATH}")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    main(random_state=args.random_state, test_size=args.test_size)
