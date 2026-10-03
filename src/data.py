"""Loading and validating the UCI Wine Quality dataset.

Nothing in this module swallows an error: every problem raises
:class:`WineDataError` so the calling Airflow task fails loudly.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from . import config

LOGGER = logging.getLogger(__name__)


class WineDataError(RuntimeError):
    """Raised when the dataset cannot be loaded or fails validation."""


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Read the wine quality CSV into a DataFrame.

    Args:
        path: Optional project-relative or absolute path. Defaults to the
            configured :data:`src.config.DATASET_PATH`.

    Returns:
        The raw dataset as a ``DataFrame``.

    Raises:
        WineDataError: If the file is missing or cannot be parsed.
    """
    dataset_path = Path(path) if path is not None else config.DATASET_PATH
    if not dataset_path.is_file():
        raise WineDataError(
            f"Wine quality dataset not found at '{dataset_path}'. "
            "Run `python -m scripts.download_data` to fetch it from the UCI repository."
        )

    try:
        dataframe = pd.read_csv(dataset_path, sep=config.DATASET_SEP)
    except Exception as exc:  # noqa: BLE001 - re-raised as a domain error below
        raise WineDataError(f"Could not parse '{dataset_path}' as a CSV file: {exc}") from exc

    LOGGER.info("Loaded %s rows x %s columns from %s", len(dataframe), len(dataframe.columns), dataset_path)
    return dataframe


def _check_columns(dataframe: pd.DataFrame) -> None:
    missing = [column for column in config.EXPECTED_COLUMNS if column not in dataframe.columns]
    if missing:
        raise WineDataError(
            f"Dataset is missing expected column(s): {missing}. Found: {list(dataframe.columns)}"
        )


def _check_numeric_features(dataframe: pd.DataFrame) -> None:
    non_numeric = [
        column for column in config.FEATURE_COLUMNS if not pd.api.types.is_numeric_dtype(dataframe[column])
    ]
    if non_numeric:
        raise WineDataError(f"Feature column(s) are not numeric: {non_numeric}")


def _check_missing_values(dataframe: pd.DataFrame) -> None:
    missing = dataframe.isna().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        raise WineDataError(f"Dataset contains missing values: {missing.to_dict()}")


def _check_row_count(dataframe: pd.DataFrame) -> None:
    n_rows = len(dataframe)
    if n_rows == 0:
        raise WineDataError("Dataset is empty.")
    if not config.MIN_ROWS <= n_rows <= config.MAX_ROWS:
        raise WineDataError(
            f"Dataset has an implausible number of rows ({n_rows}); "
            f"expected between {config.MIN_ROWS} and {config.MAX_ROWS}."
        )


def _check_target(dataframe: pd.DataFrame) -> None:
    target = dataframe[config.TARGET_COLUMN]
    unique = sorted(pd.unique(target))
    if len(unique) < 2:
        raise WineDataError(f"Target column '{config.TARGET_COLUMN}' has fewer than 2 distinct values: {unique}")
    if any(pd.isna(value) for value in unique):
        raise WineDataError(f"Target column '{config.TARGET_COLUMN}' contains NaN values.")


def describe_dataset(dataframe: pd.DataFrame) -> dict:
    """Return a JSON-serialisable summary used for logging and MLflow tags."""
    quality_counts = dataframe[config.TARGET_COLUMN].value_counts().sort_index()
    return {
        "n_rows": int(len(dataframe)),
        "n_columns": int(len(dataframe.columns)),
        "feature_columns": list(config.FEATURE_COLUMNS),
        "target_column": config.TARGET_COLUMN,
        "n_missing_values": int(dataframe.isna().sum().sum()),
        "quality_min": int(dataframe[config.TARGET_COLUMN].min()),
        "quality_max": int(dataframe[config.TARGET_COLUMN].max()),
        "quality_distribution": {str(int(k)): int(v) for k, v in quality_counts.items()},
        "quality_mean": round(float(dataframe[config.TARGET_COLUMN].mean()), 4),
    }


def validate_dataset(dataframe: pd.DataFrame) -> dict:
    """Validate a wine quality DataFrame.

    Args:
        dataframe: The raw dataset.

    Returns:
        A summary dictionary produced by :func:`describe_dataset`.

    Raises:
        WineDataError: If any validation check fails.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise WineDataError(f"Expected a pandas DataFrame, got {type(dataframe)!r}.")

    _check_row_count(dataframe)
    _check_columns(dataframe)
    _check_numeric_features(dataframe)
    _check_missing_values(dataframe)
    _check_target(dataframe)

    summary = describe_dataset(dataframe)
    LOGGER.info(
        "Dataset validated: %s rows, %s columns, quality range %s-%s, no missing values.",
        summary["n_rows"],
        summary["n_columns"],
        summary["quality_min"],
        summary["quality_max"],
    )
    LOGGER.info("quality distribution: %s", summary["quality_distribution"])
    return summary


def load_and_validate(path: str | Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Convenience wrapper used by the pipeline and the tests."""
    dataframe = load_dataset(path)
    return dataframe, validate_dataset(dataframe)


def save_summary(summary: dict, path: Path) -> None:
    """Persist a dataset summary to disk (used as an Airflow task output)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
