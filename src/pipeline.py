"""Task-level entry points called by the Airflow DAG.

The DAG stays thin: it only wires dependencies and calls the functions below.
Each function is independently runnable from a shell (`python -m src.pipeline
load_data`) which makes the same code testable without an Airflow runtime.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd

from . import config, data as data_mod, evaluate as evaluate_mod, preprocess, train as train_mod
from .tracking import setup_tracking

LOGGER = logging.getLogger("wine_quality.pipeline")

DATASET_STAGE_PATH = config.STAGE_DIR / "dataset.csv"
DATASET_SUMMARY_PATH = config.STAGE_DIR / "dataset_summary.json"
SPLITS_PATH = config.STAGE_DIR / "splits.joblib"
PREPROCESS_META_PATH = config.STAGE_DIR / "preprocess_meta.json"


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        stream=sys.stdout,
    )


def load_data() -> dict:
    """Airflow task `load_data`: read the CSV and stage a project-local copy."""
    config.ensure_directories()
    dataframe = data_mod.load_dataset()
    DATASET_STAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(DATASET_STAGE_PATH, index=False, sep=config.DATASET_SEP)
    summary = {"staged_path": str(DATASET_STAGE_PATH), "n_rows": int(len(dataframe)), "n_columns": int(len(dataframe.columns))}
    LOGGER.info("Staged %s rows x %s columns at %s", summary["n_rows"], summary["n_columns"], DATASET_STAGE_PATH)
    return summary


def validate_data() -> dict:
    """Airflow task `validate_data`: run the full validation suite."""
    if not DATASET_STAGE_PATH.is_file():
        raise data_mod.WineDataError(
            f"Staged dataset '{DATASET_STAGE_PATH}' is missing. Run the 'load_data' task first."
        )
    dataframe = data_mod.load_dataset(DATASET_STAGE_PATH)
    summary = data_mod.validate_dataset(dataframe)
    data_mod.save_summary(summary, DATASET_SUMMARY_PATH)
    return summary


def preprocess_data() -> dict:
    """Airflow task `preprocess_data`: target creation + stratified split."""
    config.ensure_directories()
    dataframe = data_mod.load_dataset()
    data_mod.validate_dataset(dataframe)

    labelled = preprocess.create_binary_target(dataframe, config.QUALITY_THRESHOLD)
    distribution = preprocess.check_target_distribution(labelled)
    x_train, x_test, y_train, y_test = preprocess.split_data(labelled)

    joblib.dump(
        {
            "x_train": x_train,
            "x_test": x_test,
            "y_train": y_train,
            "y_test": y_test,
        },
        SPLITS_PATH,
    )

    meta = {
        "quality_threshold": config.QUALITY_THRESHOLD,
        "random_state": config.RANDOM_STATE,
        "test_size": config.TEST_SIZE,
        "average_strategy": config.AVERAGE_STRATEGY,
        "splits_path": str(SPLITS_PATH),
        "n_train_rows": int(len(x_train)),
        "n_test_rows": int(len(x_test)),
        "n_features": len(config.FEATURE_COLUMNS),
        "feature_columns": list(config.FEATURE_COLUMNS),
        "class_distribution": distribution,
        "train_class_counts": {str(int(k)): int(v) for k, v in y_train.value_counts().sort_index().items()},
        "test_class_counts": {str(int(k)): int(v) for k, v in y_test.value_counts().sort_index().items()},
    }
    PREPROCESS_META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    LOGGER.info("Preprocessing complete: %s", meta["class_distribution"])
    return meta


def train_models(dag_run_id: str | None = None) -> dict:
    """Airflow task `train_models`: fit both models, one MLflow run each."""
    config.ensure_directories()
    if not SPLITS_PATH.is_file():
        raise data_mod.WineDataError(
            f"Preprocessed splits '{SPLITS_PATH}' are missing. Run the 'preprocess_data' task first."
        )
    splits = joblib.load(SPLITS_PATH)
    tags = {"pipeline": "wine_quality_pipeline"}
    if dag_run_id:
        tags["airflow_dag_run_id"] = dag_run_id

    results = train_mod.train_models(
        x_train=splits["x_train"],
        y_train=splits["y_train"],
        threshold=config.QUALITY_THRESHOLD,
        random_state=config.RANDOM_STATE,
        experiment_name=config.EXPERIMENT_NAME,
        tracking_uri=config.MLFLOW_TRACKING_URI,
        tags=tags,
    )
    return {
        model_key: {"run_id": entry["run_id"], "model_id": entry["model_id"], "model_label": entry["model_label"]}
        for model_key, entry in results.items()
    }


def evaluate_models(run_ids: dict) -> dict:
    """Airflow task `evaluate_models`: metrics + artifacts into the MLflow runs."""
    config.ensure_directories()
    if not SPLITS_PATH.is_file():
        raise data_mod.WineDataError(
            f"Preprocessed splits '{SPLITS_PATH}' are missing. Run the 'preprocess_data' task first."
        )
    if not run_ids:
        raise data_mod.WineDataError("No MLflow run ids were provided by the 'train_models' task.")

    splits = joblib.load(SPLITS_PATH)
    training_meta = train_mod.load_run_ids()
    # evaluate_pipeline() appends one section per model; start each run with a fresh file.
    config.CLASSIFICATION_REPORT_PATH.unlink(missing_ok=True)
    setup_tracking(config.MLFLOW_TRACKING_URI)

    results: dict[str, dict] = {}
    for model_key in config.MODEL_KEYS:
        if model_key not in run_ids:
            raise data_mod.WineDataError(
                f"MLflow run id for model '{model_key}' is missing from the upstream task output."
            )
        results[model_key] = evaluate_mod.evaluate_model(
            model_key=model_key,
            model_path=training_meta[model_key]["model_path"],
            x_test=splits["x_test"],
            y_test=splits["y_test"],
            run_id=run_ids[model_key]["run_id"],
            experiment_name=config.EXPERIMENT_NAME,
            tracking_uri=config.MLFLOW_TRACKING_URI,
        )

    table = evaluate_mod.format_comparison_table(results)
    evaluate_mod.save_results(results)
    config.SUMMARY_PATH.write_text(table + "\n", encoding="utf-8")
    LOGGER.info("Model comparison (test set):\n%s", table)
    return results


def run_all() -> dict:
    """Run the whole workflow in one process (used by the CLI and by smoke tests)."""
    load_data()
    validate_data()
    preprocess_data()
    run_ids = train_models()
    return evaluate_models(run_ids)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Wine quality MLOps pipeline steps")
    parser.add_argument(
        "step",
        choices=["load_data", "validate_data", "preprocess_data", "train_models", "evaluate_models", "all"],
    )
    args = parser.parse_args(argv)
    _configure_logging()
    setup_tracking(config.MLFLOW_TRACKING_URI)

    if args.step == "load_data":
        load_data()
    elif args.step == "validate_data":
        validate_data()
    elif args.step == "preprocess_data":
        preprocess_data()
    elif args.step == "train_models":
        train_models()
    elif args.step == "evaluate_models":
        evaluate_models(train_mod.load_run_ids())
    else:
        run_all()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
