"""Model training with MLflow run/parameter/model logging."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from . import config
from .data import WineDataError
from .preprocess import build_model_pipeline
from .tracking import mlflow_run, setup_tracking

LOGGER = logging.getLogger(__name__)

RUNS_METADATA_PATH = config.STAGE_DIR / "run_ids.json"


def _log_common_params(threshold: int, random_state: int, scaled: bool) -> None:
    """Parameters that are shared by every model in the experiment."""
    mlflow.log_params(
        {
            "target_column": config.TARGET_COLUMN,
            "binary_target_column": config.BINARY_TARGET_COLUMN,
            "quality_threshold": threshold,
            "random_state": random_state,
            "test_size": config.TEST_SIZE,
            "n_features": len(config.FEATURE_COLUMNS),
            "feature_scaling": "StandardScaler" if scaled else "none",
        }
    )


def train_model(
    model_key: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    threshold: int,
    random_state: int,
    experiment_name: str,
    tracking_uri: str,
    tags: dict | None = None,
) -> dict:
    """Fit one model, log it to MLflow and persist it locally.

    Returns a dict with the fitted pipeline path and the MLflow run id.
    """
    if model_key not in config.MODEL_SPECS:
        raise WineDataError(
            f"Unknown model '{model_key}'. Available models: {list(config.MODEL_SPECS)}"
        )
    spec = config.MODEL_SPECS[model_key]

    setup_tracking(tracking_uri)
    pipeline = build_model_pipeline(model_key, scale=bool(spec["scaled"]))

    LOGGER.info("Training %s ...", spec["label"])
    start = time.perf_counter()
    pipeline.fit(x_train, y_train)
    elapsed = time.perf_counter() - start
    LOGGER.info("Trained %s in %.3fs", spec["label"], elapsed)

    run_tags = {"model_type": model_key, "model_label": spec["label"], **(tags or {})}
    with mlflow_run(experiment_name, spec["label"], run_tags) as run_id:
        params = {"model_type": model_key, **spec["params"]}
        mlflow.log_params(params)
        _log_common_params(threshold, random_state, scaled=bool(spec["scaled"]))

        model_path = config.MODELS_DIR / f"{model_key}.joblib"
        joblib.dump(pipeline, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="model_file")

        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name=config.MLFLOW_MODEL_NAME,
            serialization_format="pickle",
            signature=mlflow.models.infer_signature(x_train),
            input_example=x_train.head(5),
        )
        mlflow.set_tag("training_seconds", round(elapsed, 4))
        LOGGER.info("Logged %s to MLflow run %s (model_id=%s)", spec["label"], run_id, model_info.model_id)

    return {
        "model_key": model_key,
        "model_label": spec["label"],
        "run_id": run_id,
        "model_id": model_info.model_id,
        "model_path": str(model_path),
        "training_seconds": round(elapsed, 4),
        "params": params,
    }


def train_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    threshold: int,
    random_state: int,
    experiment_name: str,
    tracking_uri: str,
    tags: dict | None = None,
) -> dict:
    """Train every configured model, each in its own MLflow run."""
    results: dict[str, dict] = {}
    for model_key in config.MODEL_KEYS:
        results[model_key] = train_model(
            model_key=model_key,
            x_train=x_train,
            y_train=y_train,
            threshold=threshold,
            random_state=random_state,
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
            tags=tags,
        )
    save_run_ids(results)
    return results


def save_run_ids(results: dict[str, dict], path: Path = RUNS_METADATA_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")


def load_run_ids(path: Path = RUNS_METADATA_PATH) -> dict[str, dict]:
    if not path.is_file():
        raise WineDataError(
            f"No training metadata found at '{path}'. Run the 'train_models' task before 'evaluate_models'."
        )
    return json.loads(path.read_text(encoding="utf-8"))
