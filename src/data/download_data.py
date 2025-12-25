# src/data/download_data.py

"""
Download the *processed* Heart Disease files directly into data/raw.

We download:
  - Processed.cleveland.data
  - Processed.hungarian.data
  - Processed.switzerland.data
  - Processed.va.data

These are the preprocessed numeric versions (with -9 as missing) that some
courses and repositories use, instead of the original raw UCI 76-column files.
"""

import argparse
from pathlib import Path
from typing import Dict

import requests

from src.config import RAW_DATA_DIR


# Base URL where the processed files are hosted.
# This is the original UCI "processed" directory:
# https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/
UCI_PROCESSED_BASE = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease"
)

PROCESSED_FILES: Dict[str, str] = {
    "processed.cleveland.data": "processed.cleveland.data",
    "processed.hungarian.data": "processed.hungarian.data",
    "processed.switzerland.data": "processed.switzerland.data",
    "processed.va.data": "processed.va.data",
}


def download_file(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"[INFO] File already exists, skipping download: {dest.name}")
        return

    print(f"[INFO] Downloading {dest.name} from {url}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)

    print(f"[OK] Downloaded to {dest}")


def download_all_raw_files(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)

    for remote_name, local_name in PROCESSED_FILES.items():
        url = f"{UCI_PROCESSED_BASE}/{remote_name}"
        dest = raw_dir / local_name
        download_file(url, dest)


def main(raw_dir: Path) -> None:
    print("Heart Disease UCI - PROCESSED numeric files")
    print("Source:", "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/")
    print(f"[INFO] Using raw data directory: {raw_dir}")
    download_all_raw_files(raw_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(RAW_DATA_DIR),
        help="Directory to store raw processed .data files (default: RAW_DATA_DIR)",
    )
    args = parser.parse_args()

    main(Path(args.raw_dir))