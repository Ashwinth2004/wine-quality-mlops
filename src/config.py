"""Central configuration for the wine quality MLOps project.

Every tunable value lives here so that both the Airflow DAG and the tests read
exactly the same settings. Values can be overridden with environment variables
(see ``.env.example``), which keeps the project free of machine specific paths.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

# --- Windows console fix -----------------------------------------------------
# MLflow prints a "run URL" banner (with a non-ASCII emoji) whenever a run ends.
# The default Windows console encoding is cp1252, which raises UnicodeEncodeError
# *inside* mlflow.end_run() and leaves runs stuck in RUNNING state. Suppress the
# banner and force UTF-8 on the standard streams.
os.environ.setdefault("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "1")
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
# -----------------------------------------------------------------------------


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw).expanduser().resolve() if raw else default


DATA_DIR = _env_path("WINE_QUALITY_DATA_DIR", PROJECT_ROOT / "data")
ARTIFACTS_DIR = _env_path("WINE_QUALITY_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts")
MODELS_DIR = _env_path("WINE_QUALITY_MODELS_DIR", PROJECT_ROOT / "models")
STAGE_DIR = ARTIFACTS_DIR / "stage"

DATASET_URL = "https://archive.ics.uci.edu/static/public/186/wine+quality.zip"
DATASET_FILENAME = os.getenv("WINE_QUALITY_CSV", "winequality-red.csv")
DATASET_PATH = _env_path("WINE_QUALITY_DATA_FILE", DATA_DIR / DATASET_FILENAME)

# UCI "winequality-red.csv" is semicolon separated, not comma separated.
DATASET_SEP = ";"

TARGET_COLUMN = "quality"
FEATURE_COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]
EXPECTED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]

# quality >= 6 -> "good". Inspecting the raw distribution shows quality 3..8 with
# 638 wines at 6, so this cut keeps the classes almost balanced (53.5% / 46.5%).
QUALITY_THRESHOLD = int(os.getenv("WINE_QUALITY_THRESHOLD", "6"))
BINARY_TARGET_COLUMN = "is_good_quality"
NEGATIVE_LABEL = "not_good"
POSITIVE_LABEL = "good"

RANDOM_STATE = int(os.getenv("WINE_QUALITY_RANDOM_STATE", "42"))
TEST_SIZE = float(os.getenv("WINE_QUALITY_TEST_SIZE", "0.2"))

MIN_ROWS = 100
MAX_ROWS = 1_000_000
MIN_POSITIVE_CLASS_FRACTION = 0.05
MAX_POSITIVE_CLASS_FRACTION = 0.95

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "wine-quality-classification")
MLFLOW_MODEL_NAME = "model"

# Average strategy for precision / recall / f1. `weighted` is used because the
# binarised target is close to balanced, so per-class scores are not skewed and a
# single number is enough to rank the two models.
AVERAGE_STRATEGY = "weighted"

LOGISTIC_REGRESSION_PARAMS = {
    "C": float(os.getenv("LOGREG_C", "1.0")),
    "max_iter": int(os.getenv("LOGREG_MAX_ITER", "1000")),
    "random_state": RANDOM_STATE,
}
RANDOM_FOREST_PARAMS = {
    "n_estimators": int(os.getenv("RF_N_ESTIMATORS", "300")),
    "max_depth": int(os.getenv("RF_MAX_DEPTH", "10")),
    "random_state": RANDOM_STATE,
}

MODEL_SPECS = {
    "logistic_regression": {
        "label": "Logistic Regression",
        "scaled": True,
        "params": LOGISTIC_REGRESSION_PARAMS,
    },
    "random_forest": {
        "label": "Random Forest",
        "scaled": False,
        "params": RANDOM_FOREST_PARAMS,
    },
}
MODEL_KEYS = list(MODEL_SPECS)

CLASSIFICATION_REPORT_PATH = ARTIFACTS_DIR / "classification_report.txt"
CONFUSION_MATRIX_DIR = ARTIFACTS_DIR / "confusion_matrices"
RESULTS_PATH = ARTIFACTS_DIR / "results.json"
SUMMARY_PATH = ARTIFACTS_DIR / "summary.txt"


def ensure_directories() -> None:
    """Create every project directory the pipeline writes to."""
    for directory in (DATA_DIR, ARTIFACTS_DIR, MODELS_DIR, STAGE_DIR, CONFUSION_MATRIX_DIR):
        directory.mkdir(parents=True, exist_ok=True)
