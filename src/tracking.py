"""Thin MLflow helper layer shared by training and evaluation.

Keeps tracking-server configuration in one place so the Airflow tasks and the
local scripts always talk to the same MLflow instance.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from collections.abc import Iterator

import mlflow
from mlflow.tracking import MlflowClient

from . import config

LOGGER = logging.getLogger(__name__)


def setup_tracking(tracking_uri: str | None = None) -> str:
    """Point MLflow at the local tracking server and return the URI used."""
    uri = tracking_uri or config.MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(uri)
    LOGGER.info("MLflow tracking URI: %s", mlflow.get_tracking_uri())
    return mlflow.get_tracking_uri()


def get_or_create_experiment(experiment_name: str) -> str:
    """Return the experiment id, creating the experiment when it does not exist."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(experiment_name)
        LOGGER.info("Created MLflow experiment '%s' (id=%s)", experiment_name, experiment_id)
    else:
        experiment_id = experiment.experiment_id
        LOGGER.info("Reusing MLflow experiment '%s' (id=%s)", experiment_name, experiment_id)
    return experiment_id


@contextmanager
def mlflow_run(experiment_name: str, run_name: str, tags: dict | None = None) -> Iterator[str]:
    """Start an active MLflow run inside ``experiment_name`` and always close it.

    If the body raises, the run is marked ``FAILED`` instead of being left open.
    """
    experiment_id = get_or_create_experiment(experiment_name)
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, tags=tags or {}) as run:
        try:
            yield run.info.run_id
        except Exception:
            mlflow.set_tag("run_status_note", "exception raised inside mlflow_run")
            raise
