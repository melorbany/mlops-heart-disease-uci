import argparse
from pathlib import Path
from typing import List

import pandas as pd

from src.config import RAW_DATA_DIR

COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]


def load_processed_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLUMNS)
    return df


def convert_all(inputs: List[Path], site_names: List[str], output_path: Path):
    dfs = []
    for p, site in zip(inputs, site_names):
        if not p.exists():
            print(f"[WARN] missing file, skipping: {p}")
            continue
        print(f"[INFO] loading {p} as site='{site}'")
        dfs.append(load_processed_file(p))

    if not dfs:
        raise FileNotFoundError("No processed.*.data files found.")

    df = pd.concat(dfs, ignore_index=True)

    # Binary target
    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"[OK] saved merged CSV to {output_path}")
    print(f"     shape: {df.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default=str(RAW_DATA_DIR / "heart.csv"),
    )
    args = parser.parse_args()

    base = RAW_DATA_DIR
    inputs = [
        base / "processed.cleveland.data",
        base / "processed.hungarian.data",
        base / "processed.switzerland.data",
        base / "processed.va.data",
    ]
    sites = ["cleveland", "hungarian", "switzerland", "va"]

    convert_all(inputs, sites, Path(args.output))


if __name__ == "__main__":
    main()