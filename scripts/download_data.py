"""Download the UCI Wine Quality dataset into ``data/``.

Usage::

    python -m scripts.download_data

The archive is fetched from the official UCI Machine Learning Repository
(https://archive.ics.uci.edu/dataset/186/wine+quality) and only the two CSV
files are extracted. Existing files are kept unless ``--force`` is passed.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from src import config

LOGGER = logging.getLogger(__name__)

WANTED_FILES = ("winequality-red.csv", "winequality-white.csv")


def download_dataset(destination: Path | None = None, force: bool = False) -> list[Path]:
    """Download and extract the dataset. Returns the list of extracted paths."""
    destination = destination or config.DATA_DIR
    destination.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for filename in WANTED_FILES:
        target = destination / filename
        if target.is_file() and not force:
            LOGGER.info("%s already exists, skipping download.", target)
            written.append(target)

    if len(written) == len(WANTED_FILES):
        return written

    LOGGER.info("Downloading %s", config.DATASET_URL)
    with urllib.request.urlopen(config.DATASET_URL) as response:  # noqa: S310 - fixed, trusted URL
        payload = response.read()

    with tempfile.TemporaryDirectory() as tmp_dir:
        archive_path = Path(tmp_dir) / "wine-quality.zip"
        archive_path.write_bytes(payload)
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            LOGGER.info("Archive contains: %s", names)
            for filename in WANTED_FILES:
                member = next((n for n in names if n.endswith(filename)), None)
                if member is None:
                    raise FileNotFoundError(f"'{filename}' not found inside the downloaded archive.")
                target = destination / filename
                if target.is_file() and not force:
                    LOGGER.info("%s already exists, keeping it.", target)
                else:
                    with archive.open(member) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    LOGGER.info("Wrote %s (%s bytes)", target, target.stat().st_size)
                written.append(target)

    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download even if the files exist.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    paths = download_dataset(force=args.force)
    LOGGER.info("Dataset ready: %s", [str(p) for p in paths])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
