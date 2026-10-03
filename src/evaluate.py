"""Evaluation: metrics, confusion matrix, classification report, MLflow logging."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from . import config
from .data import WineDataError
from .tracking import setup_tracking

LOGGER = logging.getLogger(__name__)

METRIC_NAMES = ("accuracy", "precision", "recall", "f1_score")

matplotlib = None


def _pyplot():
    """Import matplotlib lazily with a non-interactive backend."""
    global matplotlib
    if matplotlib is None:
        import matplotlib as mpl

        mpl.use("Agg")
        from matplotlib import pyplot

        matplotlib = (mpl, pyplot)
    return matplotlib


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """Compute accuracy / precision / recall / f1 on the given split."""
    average = config.AVERAGE_STRATEGY
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }
    for name, value in metrics.items():
        if not 0.0 <= value <= 1.0:
            raise WineDataError(f"Metric '{name}' is out of the [0, 1] range: {value}")
    return metrics


def plot_confusion_matrix(
    y_true: pd.Series, y_pred: np.ndarray, model_label: str, output_path: Path
) -> Path:
    """Render the confusion matrix to ``output_path`` and close the figure."""
    _, pyplot = _pyplot()
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    figure, axes = pyplot.subplots(figsize=(5, 4.5))
    image = axes.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axes)
    labels = [config.NEGATIVE_LABEL, config.POSITIVE_LABEL]
    axes.set(
        xticks=range(2),
        yticks=range(2),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted label",
        ylabel="True label",
        title=f"Confusion matrix - {model_label}",
    )
    axes.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2.0
    for row in range(2):
        for col in range(2):
            axes.text(
                col,
                row,
                format(matrix[row, col], "d"),
                ha="center",
                va="center",
                color="white" if matrix[row, col] > threshold else "black",
            )
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=120)
    pyplot.close(figure)
    LOGGER.info("Saved confusion matrix to %s", output_path)
    return output_path


def build_classification_report(y_true: pd.Series, y_pred: np.ndarray, model_label: str) -> str:
    return (
        f"Model: {model_label}\n"
        f"Averaging strategy: {config.AVERAGE_STRATEGY}\n"
        f"Labels: 0={config.NEGATIVE_LABEL}, 1={config.POSITIVE_LABEL}\n\n"
        f"{classification_report(y_true, y_pred, labels=[0, 1], target_names=[config.NEGATIVE_LABEL, config.POSITIVE_LABEL], zero_division=0)}"
    )


def evaluate_pipeline(
    pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    model_label: str,
) -> dict:
    """Predict on the test set and return metrics plus artifact paths."""
    predictions = pipeline.predict(x_test)
    if len(predictions) != len(y_test):
        raise WineDataError(
            f"{model_label} produced {len(predictions)} predictions for {len(y_test)} rows."
        )

    metrics = compute_metrics(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    figure_path = config.CONFUSION_MATRIX_DIR / f"{model_label.lower().replace(' ', '_')}.png"
    plot_confusion_matrix(y_test, predictions, model_label, figure_path)

    report_text = build_classification_report(y_test, predictions, model_label)
    report_path = config.CLASSIFICATION_REPORT_PATH
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("a", encoding="utf-8") as handle:
        handle.write(report_text + "\n" + "-" * 72 + "\n")

    LOGGER.info(
        "%s test metrics -> accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f",
        model_label,
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"],
    )
    return {
        "model_label": model_label,
        "metrics": metrics,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_artifact": str(figure_path),
        "classification_report_artifact": str(report_path),
        "classification_report_text": report_text,
    }


def evaluate_model(
    model_key: str,
    model_path: str | Path,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    run_id: str,
    experiment_name: str,
    tracking_uri: str,
) -> dict:
    """Load a persisted pipeline, evaluate it and log everything to its MLflow run."""
    spec = config.MODEL_SPECS.get(model_key)
    if spec is None:
        raise WineDataError(f"Unknown model '{model_key}'.")
    model_label = spec["label"]

    path = Path(model_path)
    if not path.is_file():
        raise WineDataError(f"Trained model not found at '{path}'.")

    pipeline = joblib.load(path)
    result = evaluate_pipeline(pipeline, x_test, y_test, model_label)

    setup_tracking(tracking_uri)
    metrics = result["metrics"]
    mlflow.log_metrics(metrics, run_id=run_id)
    # log_text() takes the report *content*; log_artifact() takes a file path.
    mlflow.log_text(result["classification_report_text"], "classification_report.txt", run_id=run_id)
    mlflow.log_artifact(result["confusion_matrix_artifact"], "confusion_matrix.png", run_id=run_id)
    MlflowClient().set_tag(run_id, "evaluated", "true")
    LOGGER.info("Logged metrics and artifacts to MLflow run %s", run_id)

    result["run_id"] = run_id
    result["model_key"] = model_key
    result["experiment_name"] = experiment_name
    result.pop("classification_report_text")
    return result


def format_comparison_table(results: dict[str, dict]) -> str:
    """Render the model comparison table used in logs, files and the README."""
    header = (
        f"{'Model':<22}{'Accuracy':>12}{'Precision':>12}{'Recall':>10}{'F1':>10}"
    )
    separator = "-" * len(header)
    lines = [header, separator]
    for model_key in config.MODEL_KEYS:
        if model_key not in results:
            continue
        entry = results[model_key]
        metrics = entry["metrics"]
        lines.append(
            f"{entry['model_label']:<22}"
            f"{metrics['accuracy']:>12.4f}"
            f"{metrics['precision']:>12.4f}"
            f"{metrics['recall']:>10.4f}"
            f"{metrics['f1_score']:>10.4f}"
        )
    return "\n".join(lines)


def save_results(results: dict[str, dict], path: Path = config.RESULTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
