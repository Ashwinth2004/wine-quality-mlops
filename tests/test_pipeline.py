"""Tests for data, target, preprocessing, models, metrics, MLflow logging and the DAG."""

from __future__ import annotations

import sys
from pathlib import Path

import mlflow
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config, data, evaluate, preprocess, train  # noqa: E402


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return data.load_dataset()


@pytest.fixture(scope="module")
def labelled(raw) -> pd.DataFrame:
    return preprocess.create_binary_target(raw, config.QUALITY_THRESHOLD)


@pytest.fixture(scope="module")
def splits(labelled):
    return preprocess.split_data(labelled)


# --- data ---------------------------------------------------------------------
def test_dataset_loads_and_is_non_empty(raw):
    assert len(raw) > 0


def test_expected_columns_exist(raw):
    assert set(config.EXPECTED_COLUMNS) <= set(raw.columns)


def test_validation_passes_on_real_data(raw):
    summary = data.validate_dataset(raw)
    assert summary["n_missing_values"] == 0


def test_validation_rejects_bad_data(raw):
    with pytest.raises(data.WineDataError):
        data.validate_dataset(raw.iloc[0:0])
    with pytest.raises(data.WineDataError):
        data.validate_dataset(raw.drop(columns=["alcohol"]))
    broken = raw.copy()
    broken.loc[0, "pH"] = None
    with pytest.raises(data.WineDataError):
        data.validate_dataset(broken)
    text = raw.copy()
    text["pH"] = text["pH"].astype(str)
    with pytest.raises(data.WineDataError):
        data.validate_dataset(text)


def test_missing_file_raises(tmp_path):
    with pytest.raises(data.WineDataError):
        data.load_dataset(tmp_path / "nope.csv")


# --- target -------------------------------------------------------------------
def test_binary_target_created(raw, labelled):
    assert set(labelled[config.BINARY_TARGET_COLUMN].unique()) == {0, 1}
    assert config.BINARY_TARGET_COLUMN not in raw.columns  # input untouched
    expected = (raw[config.TARGET_COLUMN] >= config.QUALITY_THRESHOLD).astype(int)
    assert labelled[config.BINARY_TARGET_COLUMN].equals(expected)
    assert labelled[config.TARGET_COLUMN].equals(raw[config.TARGET_COLUMN])


def test_target_not_collapsed(labelled):
    assert preprocess.check_target_distribution(labelled)["is_balanced"]
    with pytest.raises(data.WineDataError):
        preprocess.check_target_distribution(preprocess.create_binary_target(labelled, 99))


# --- preprocessing ------------------------------------------------------------
def test_split_is_stratified_and_disjoint(labelled, splits):
    x_train, x_test, y_train, y_test = splits
    assert len(x_train) + len(x_test) == len(labelled)
    assert set(x_train.index).isdisjoint(x_test.index)
    assert list(x_train.columns) == config.FEATURE_COLUMNS
    assert abs(y_train.mean() - y_test.mean()) < 0.02


def test_scaler_fitted_on_train_only(splits):
    x_train, x_test, y_train, _ = splits
    pipe = preprocess.build_model_pipeline("logistic_regression", scale=True).fit(x_train, y_train)
    scaler = pipe.named_steps["preprocessor"].named_transformers_["numeric"].named_steps["scaler"]
    assert scaler.mean_ == pytest.approx(x_train.mean().to_numpy())
    assert scaler.n_samples_seen_ == len(x_train)


# --- models + metrics ---------------------------------------------------------
@pytest.mark.parametrize("model_key", config.MODEL_KEYS)
def test_model_trains_predicts_and_metrics_valid(model_key, splits):
    x_train, x_test, y_train, y_test = splits
    pipe = preprocess.build_model_pipeline(model_key, scale=config.MODEL_SPECS[model_key]["scaled"])
    pipe.fit(x_train, y_train)
    predictions = pipe.predict(x_test)
    assert len(predictions) == len(y_test)
    assert set(predictions) <= {0, 1}
    metrics = evaluate.compute_metrics(y_test, predictions)
    assert set(metrics) == set(evaluate.METRIC_NAMES)
    assert all(0.0 <= v <= 1.0 for v in metrics.values())
    assert metrics["accuracy"] > 0.5  # better than chance


def test_confusion_matrix_plot_written(tmp_path):
    out = evaluate.plot_confusion_matrix(pd.Series([0, 1, 1, 0]), [0, 1, 0, 0], "Test", tmp_path / "cm.png")
    assert out.is_file() and out.stat().st_size > 0


# --- MLflow integration (isolated sqlite backend, no server needed) -----------
def test_train_and_evaluate_log_to_mlflow(tmp_path, splits, monkeypatch):
    x_train, x_test, y_train, y_test = splits
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(train.config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "CONFUSION_MATRIX_DIR", tmp_path / "cm")
    monkeypatch.setattr(config, "CLASSIFICATION_REPORT_PATH", tmp_path / "report.txt")
    (tmp_path / "models").mkdir()
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    mlflow.create_experiment("pytest-exp", artifact_location=(tmp_path / "art").as_uri())

    results = train.train_models(
        x_train, y_train, config.QUALITY_THRESHOLD, config.RANDOM_STATE, "pytest-exp", uri
    )
    assert set(results) == set(config.MODEL_KEYS)
    for key, entry in results.items():
        evaluate.evaluate_model(key, entry["model_path"], x_test, y_test, entry["run_id"], "pytest-exp", uri)

    client = MlflowClient()
    for key, entry in results.items():
        run = client.get_run(entry["run_id"])
        assert run.info.status == "FINISHED"
        assert run.data.params["model_type"] == key
        assert {"accuracy", "precision", "recall", "f1_score"} <= set(run.data.metrics)
        names = {a.path for a in client.list_artifacts(entry["run_id"])}
        assert {"classification_report.txt", "confusion_matrix.png"} <= names
        assert client.search_logged_models(experiment_ids=[run.info.experiment_id])


# --- Airflow ------------------------------------------------------------------
def test_dag_imports_and_has_expected_structure():
    try:
        from airflow.dag_processing.dagbag import DagBag
        from airflow.sdk import dag  # noqa: F401
    except (ImportError, AttributeError) as exc:  # Airflow cannot be imported on native Windows
        pytest.skip(f"Airflow is not importable on this platform (use Linux/WSL2/Docker): {exc}")

    bag = DagBag(dag_folder=str(ROOT / "dags"), bundle_path=ROOT / "dags", bundle_name="tests")
    assert not bag.import_errors, bag.import_errors
    dag = bag.get_dag("wine_quality_pipeline")
    assert dag is not None
    assert [t.task_id for t in dag.topological_sort()] == [
        "load_data", "validate_data", "preprocess_data", "train_models", "evaluate_models",
    ]
