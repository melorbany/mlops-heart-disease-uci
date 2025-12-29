import argparse
import json
from pathlib import Path

import matplotlib
import joblib
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

from src.config import (
    ARTIFACTS_DIR,
    FINAL_MODEL_PATH,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.features.build_features import build_preprocessor, split_features_target
from src.models.mlflow_utils import log_artifact, log_metrics, log_params, start_run

matplotlib.use("Agg")  # Use non‑interactive backend, avoids Tk / GUI issues


def evaluate_model(model, X_test, y_test, prefix: str, artifact_dir: Path):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        f"{prefix}_accuracy": float(accuracy_score(y_test, y_pred)),
        f"{prefix}_precision": float(precision_score(y_test, y_pred)),
        f"{prefix}_recall": float(recall_score(y_test, y_pred)),
        f"{prefix}_f1": float(f1_score(y_test, y_pred)),
        f"{prefix}_roc_auc": float(roc_auc_score(y_test, y_prob)),
    }

    # Confusion Matrix
    fig_cm, ax_cm = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax_cm)
    cm_path = artifact_dir / f"{prefix}_confusion_matrix.png"
    fig_cm.savefig(cm_path)
    plt.close(fig_cm)

    # ROC Curve
    fig_roc, ax_roc = plt.subplots()
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax_roc)
    roc_path = artifact_dir / f"{prefix}_roc_curve.png"
    fig_roc.savefig(roc_path)
    plt.close(fig_roc)

    return metrics, str(cm_path), str(roc_path)


def main(random_state: int = RANDOM_STATE, test_size: float = TEST_SIZE):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

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

    results = {}
    best_model_name = None
    best_auc = -np.inf
    best_model = None

    for name, clf in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("clf", clf),
            ]
        )

        with start_run(run_name=name):
            log_params(
                {
                    "model_name": name,
                    "random_state": random_state,
                    "test_size": test_size,
                    **clf.get_params(),
                }
            )

            # ---- Robust cross-validation ----
            n_samples = len(X_train)
            # Cap at 5 folds, but cannot exceed number of training samples
            n_splits = min(5, n_samples)

            if n_splits <= 2:
                # Not enough data for CV; skip CV and just fit once
                cv_scores = None
                cv_mean = None
                cv_std = None
                pipeline.fit(X_train, y_train)
            else:
                cv = StratifiedKFold(
                    n_splits=n_splits, shuffle=True, random_state=random_state
                )
                cv_scores = cross_val_score(
                    pipeline, X_train, y_train, cv=cv, scoring="roc_auc"
                )
                cv_mean = float(np.mean(cv_scores))
                cv_std = float(np.std(cv_scores))

                pipeline.fit(X_train, y_train)

            # Evaluate
            metrics, cm_path, roc_path = evaluate_model(
                pipeline, X_test, y_test, prefix=name, artifact_dir=ARTIFACTS_DIR
            )

            # Add CV metrics if available
            if cv_mean is not None:
                metrics[f"{name}_cv_roc_auc_mean"] = cv_mean
            if cv_std is not None:
                metrics[f"{name}_cv_roc_auc_std"] = cv_std

            log_metrics(metrics)
            log_artifact(cm_path)
            log_artifact(roc_path)

            results[name] = metrics

            if metrics[f"{name}_roc_auc"] > best_auc:
                best_auc = metrics[f"{name}_roc_auc"]
                best_model_name = name
                best_model = pipeline

    # Save metrics to JSON
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=4)

    # Save best model
    FINAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, FINAL_MODEL_PATH)

    print(f"Best model: {best_model_name} with ROC-AUC {best_auc:.4f}")
    print(f"Saved final model to {FINAL_MODEL_PATH}")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    main(random_state=args.random_state, test_size=args.test_size)
