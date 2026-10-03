"""Target creation, train/test splitting and scikit-learn preprocessing.

The split happens *before* any fitting, and every transformer is built inside a
:class:`sklearn.pipeline.Pipeline`, so nothing is ever fitted on the test set.
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .data import WineDataError

LOGGER = logging.getLogger(__name__)


def create_binary_target(
    dataframe: pd.DataFrame, threshold: int | None = None
) -> pd.DataFrame:
    """Add the binary ``is_good_quality`` column without touching ``quality``.

    ``quality >= threshold`` -> 1 (``good``), otherwise 0 (``not_good``).
    """
    threshold = config.QUALITY_THRESHOLD if threshold is None else threshold

    if config.TARGET_COLUMN not in dataframe.columns:
        raise WineDataError(f"Target column '{config.TARGET_COLUMN}' is missing.")
    if not pd.api.types.is_numeric_dtype(dataframe[config.TARGET_COLUMN]):
        raise WineDataError(f"Target column '{config.TARGET_COLUMN}' is not numeric.")

    result = dataframe.copy()
    result[config.BINARY_TARGET_COLUMN] = (
        result[config.TARGET_COLUMN] >= threshold
    ).astype(int)
    return result


def target_distribution(dataframe: pd.DataFrame) -> dict:
    """Class counts and the positive fraction of the binary target."""
    if config.BINARY_TARGET_COLUMN not in dataframe.columns:
        raise WineDataError(
            f"Binary target '{config.BINARY_TARGET_COLUMN}' is missing; "
            "call create_binary_target() first."
        )
    counts = dataframe[config.BINARY_TARGET_COLUMN].value_counts().sort_index()
    n_rows = len(dataframe)
    return {
        "0_not_good": int(counts.get(0, 0)),
        "1_good": int(counts.get(1, 0)),
        "positive_fraction": round(float(counts.get(1, 0)) / n_rows, 4) if n_rows else 0.0,
        "is_balanced": bool(
            config.MIN_POSITIVE_CLASS_FRACTION
            <= (counts.get(1, 0) / n_rows if n_rows else 0.0)
            <= config.MAX_POSITIVE_CLASS_FRACTION
        ),
    }


def check_target_distribution(dataframe: pd.DataFrame) -> dict:
    """Validate the binarised target and return its distribution summary."""
    distribution = target_distribution(dataframe)
    if distribution["0_not_good"] == 0 or distribution["1_good"] == 0:
        raise WineDataError(
            "Binary target collapsed to a single class: "
            f"{distribution}. Adjust WINE_QUALITY_THRESHOLD."
        )
    if not distribution["is_balanced"]:
        LOGGER.warning(
            "Binary target is imbalanced (positive fraction %.2f). "
            "Consider a different WINE_QUALITY_THRESHOLD.",
            distribution["positive_fraction"],
        )
    LOGGER.info(
        "Binary target: %s good / %s not_good (positive fraction %.4f)",
        distribution["1_good"],
        distribution["0_not_good"],
        distribution["positive_fraction"],
    )
    return distribution


def split_data(
    dataframe: pd.DataFrame, test_size: float | None = None, random_state: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split of features and the binary target."""
    test_size = config.TEST_SIZE if test_size is None else test_size
    random_state = config.RANDOM_STATE if random_state is None else random_state

    if config.BINARY_TARGET_COLUMN not in dataframe.columns:
        raise WineDataError(
            f"Binary target '{config.BINARY_TARGET_COLUMN}' is missing; "
            "call create_binary_target() first."
        )

    features = dataframe[config.FEATURE_COLUMNS]
    target = dataframe[config.BINARY_TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state, stratify=target
    )
    LOGGER.info(
        "Train/test split: %s train rows, %s test rows (test_size=%s, random_state=%s, stratified)",
        len(x_train),
        len(x_test),
        test_size,
        random_state,
    )
    return x_train, x_test, y_train, y_test


def build_preprocessor(features: list[str] | None = None, scale: bool = False) -> ColumnTransformer:
    """Build a ``ColumnTransformer``: median imputation, optional standardisation.

    Imputation is fitted inside the pipeline, therefore on training data only.
    """
    features = list(config.FEATURE_COLUMNS) if features is None else features
    steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(transformers=[("numeric", Pipeline(steps), features)])


def build_model_pipeline(model_key: str, scale: bool) -> Pipeline:
    """Return the full ``preprocess -> classifier`` pipeline for a model key."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    if model_key not in config.MODEL_SPECS:
        raise WineDataError(
            f"Unknown model '{model_key}'. Available models: {list(config.MODEL_SPECS)}"
        )

    spec = config.MODEL_SPECS[model_key]
    if model_key == "logistic_regression":
        classifier = LogisticRegression(**spec["params"])
    elif model_key == "random_forest":
        classifier = RandomForestClassifier(**spec["params"])
    else:  # pragma: no cover - guarded by the check above
        raise WineDataError(f"No classifier implemented for '{model_key}'.")

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale=scale)),
            ("classifier", classifier),
        ]
    )
