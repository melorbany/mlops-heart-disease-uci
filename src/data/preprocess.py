import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    TARGET_COLUMN,
)


def preprocess(input_path: Path, output_path: Path, test_size: float = 0.2):
    df = pd.read_csv(input_path)

    # Basic cleanup: drop rows with missing target
    df = df.dropna(subset=[TARGET_COLUMN])

    # Optionally handle missing values here (simple example: drop)
    df = df.dropna()

    # Save full clean dataset (no features/target split yet)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # Also optional: create train/test CSVs
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df[TARGET_COLUMN])
    train_df.to_csv(output_path.with_name("heart_train.csv"), index=False)
    test_df.to_csv(output_path.with_name("heart_test.csv"), index=False)

    print(f"[OK] saved cleaned dataset to {output_path}")
    print(f"     train: {train_df.shape}, test: {test_df.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default=str(RAW_DATA_DIR / "heart.csv"),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROCESSED_DATA_DIR / "heart_clean.csv"),
    )
    args = parser.parse_args()

    preprocess(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()