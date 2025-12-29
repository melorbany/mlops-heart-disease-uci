import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd

from src.config import PROCESSED_DATA_PATH, RAW_DATA_PATH, TARGET_COLUMN


def load_raw_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Basic cleaning logic. Adjust as necessary based on actual dataset.
    # 1. Drop duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    # 2. Handle missing values: simple strategy (fill with median/most frequent)
    for col in df.columns:
        if df[col].dtype != "object":
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    # 3. Ensure target column exists
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in data.")

    return df


def save_clean_data(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_preprocessing(input_path: Path, output_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_raw = load_raw_data(input_path)
    df_clean = clean_data(df_raw)
    save_clean_data(df_clean, output_path)
    return df_raw, df_clean


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(RAW_DATA_PATH))
    parser.add_argument("--output", type=str, default=str(PROCESSED_DATA_PATH))
    args = parser.parse_args()

    run_preprocessing(Path(args.input), Path(args.output))
    print(f"Cleaned data saved to {args.output}")
