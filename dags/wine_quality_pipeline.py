"""Airflow DAG: wine quality classification, tracked with MLflow.

The DAG only orchestrates. All machine-learning code lives in ``src/`` and all
experiment tracking is done by MLflow from inside the ``train_models`` and
``evaluate_models`` tasks.

    load_data -> validate_data -> preprocess_data -> train_models -> evaluate_models
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import pendulum
from airflow.sdk import dag, get_current_context, task

# Make ``src`` importable no matter where Airflow launches the task from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_ARGS = {
    "owner": "mlops-student",
    "retries": 1,
    "retry_delay": timedelta(seconds=30),
}


@dag(
    dag_id="wine_quality_pipeline",
    description="Load, validate, preprocess, train and evaluate wine quality models; log to MLflow.",
    schedule="@daily",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    is_paused_upon_creation=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["mlops", "mlflow", "wine-quality"],
)
def wine_quality_pipeline():
    @task
    def load_data() -> dict:
        from src import pipeline

        return pipeline.load_data()

    @task
    def validate_data(_upstream: dict) -> dict:
        from src import pipeline

        return pipeline.validate_data()

    @task
    def preprocess_data(_upstream: dict) -> dict:
        from src import pipeline

        return pipeline.preprocess_data()

    @task
    def train_models(_upstream: dict) -> dict:
        from src import pipeline

        context = get_current_context()
        return pipeline.train_models(dag_run_id=context["run_id"])

    @task
    def evaluate_models(run_ids: dict) -> dict:
        from src import pipeline

        results = pipeline.evaluate_models(run_ids)
        # Keep the XCom small: metrics only.
        return {key: entry["metrics"] for key, entry in results.items()}

    loaded = load_data()
    validated = validate_data(loaded)
    preprocessed = preprocess_data(validated)
    trained = train_models(preprocessed)
    evaluate_models(trained)


wine_quality_pipeline()
