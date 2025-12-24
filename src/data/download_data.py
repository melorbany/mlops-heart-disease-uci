"""
Download or place the Heart Disease UCI dataset.

For now, this script just documents the dataset source and expects you
to manually download and place 'heart.csv' in data/raw.

You can extend this to programmatically download from a URL if available.
"""

import argparse
from pathlib import Path
from src.config import RAW_DATA_PATH

SOURCE_URL = "https://archive.ics.uci.edu/dataset/45/heart+disease"

def main(output: Path):
    print("Please download the Heart Disease dataset from:")
    print(SOURCE_URL)
    print(f"Then save the CSV as: {output}")
    print("You can rename the downloaded file to 'heart.csv'.")
    if not output.exists():
        print(f"[INFO] Currently, file does not exist at {output}.")
    else:
        print(f"[OK] Found existing file at {output}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=str(RAW_DATA_PATH))
    args = parser.parse_args()
    main(Path(args.output))