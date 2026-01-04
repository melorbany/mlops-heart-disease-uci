import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe backend; must come before pyplot

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import ARTIFACTS_DIR, NUMERIC_FEATURES, PROCESSED_DATA_PATH, TARGET_COLUMN


def create_eda_directory(base_dir: Path) -> Path:
    eda_dir = base_dir / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)
    return eda_dir


def plot_histograms(df: pd.DataFrame, output_dir: Path) -> Path:
    """Generate histograms for all numeric features."""
    numeric_cols = [col for col in NUMERIC_FEATURES if col in df.columns]

    n_cols = 3
    n_rows = max(1, (len(numeric_cols) + n_cols - 1) // n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    # Robustly handle axes returned as Axes, 1D ndarray, or 2D ndarray
    axes = np.array(axes).ravel()

    for idx, col in enumerate(numeric_cols):
        axes[idx].hist(df[col].dropna(), bins=30, edgecolor="black", alpha=0.7)
        axes[idx].set_title(f"Distribution of {col}", fontsize=12)
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel("Frequency")
        axes[idx].grid(axis="y", alpha=0.3)

    # Hide unused subplots
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    output_path = output_dir / "histograms.png"
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Histograms saved to {output_path}")
    return output_path


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> Path:
    """Generate correlation heatmap for all numeric features including target."""
    numeric_cols = [col for col in NUMERIC_FEATURES if col in df.columns]

    # Include target column in correlation analysis
    if TARGET_COLUMN in df.columns:
        numeric_cols_with_target = numeric_cols + [TARGET_COLUMN]
    else:
        numeric_cols_with_target = numeric_cols

    correlation_matrix = df[numeric_cols_with_target].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, pad=20)

    plt.tight_layout()
    output_path = output_dir / "correlation_heatmap.png"
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Correlation heatmap saved to {output_path}")
    return output_path


def plot_class_balance(df: pd.DataFrame, output_dir: Path) -> Path:
    """Generate class balance plot for the target variable."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataframe.")

    class_counts = df[TARGET_COLUMN].value_counts().sort_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar plot
    colors = ["#2ecc71", "#e74c3c"]
    class_counts.plot(kind="bar", ax=ax1, color=colors, edgecolor="black", alpha=0.8)
    ax1.set_title("Class Distribution (Count)", fontsize=14)
    ax1.set_xlabel("Class")
    ax1.set_ylabel("Count")
    ax1.set_xticklabels(["No Disease (0)", "Disease (1)"], rotation=0)
    ax1.grid(axis="y", alpha=0.3)

    # Add count labels on bars
    for i, v in enumerate(class_counts):
        ax1.text(i, v + 5, str(v), ha="center", va="bottom", fontweight="bold")

    # Pie chart
    ax2.pie(
        class_counts,
        labels=["No Disease (0)", "Disease (1)"],
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        explode=(0.05, 0.05),
    )
    ax2.set_title("Class Distribution (Percentage)", fontsize=14)

    plt.tight_layout()
    output_path = output_dir / "class_balance.png"
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Class balance plot saved to {output_path}")
    return output_path


def generate_eda_summary(df: pd.DataFrame, output_dir: Path) -> Path:
    """Generate a text summary of the dataset."""
    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append("HEART DISEASE DATASET - EDA SUMMARY")
    summary_lines.append("=" * 60)
    summary_lines.append("")

    summary_lines.append(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    summary_lines.append("")

    summary_lines.append("Target distribution:")
    if TARGET_COLUMN in df.columns:
        class_counts = df[TARGET_COLUMN].value_counts().sort_index()
        for cls, count in class_counts.items():
            pct = (count / len(df)) * 100
            summary_lines.append(f"  Class {cls}: {count} ({pct:.1f}%)")
    summary_lines.append("")

    summary_lines.append("Missing values per column:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        summary_lines.append("  No missing values found")
    else:
        for col, count in missing[missing > 0].items():
            summary_lines.append(f"  {col}: {count}")
    summary_lines.append("")

    summary_lines.append("Numeric features summary:")
    numeric_cols = [col for col in NUMERIC_FEATURES if col in df.columns]
    for col in numeric_cols:
        stats = df[col].describe()
        summary_lines.append(
            f"  {col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, "
            f"min={stats['min']:.2f}, max={stats['max']:.2f}"
        )

    summary_lines.append("")
    summary_lines.append("=" * 60)

    output_path = output_dir / "eda_summary.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print(f"[OK] EDA summary saved to {output_path}")
    return output_path


def main(data_path: Path, output_base_dir: Path) -> None:
    """Run all EDA visualizations."""
    print(f"[INFO] Reading data from {data_path}")
    df = pd.read_csv(data_path, na_values="?")

    print(f"[INFO] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    eda_dir = create_eda_directory(output_base_dir)
    print(f"[INFO] Saving plots to {eda_dir}")

    # Generate all visualizations
    plot_histograms(df, eda_dir)
    plot_correlation_heatmap(df, eda_dir)
    plot_class_balance(df, eda_dir)
    generate_eda_summary(df, eda_dir)

    print("\n[SUCCESS] All EDA visualizations completed!")
    print(f"[INFO] Outputs saved to: {eda_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate EDA visualizations for Heart Disease dataset"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(PROCESSED_DATA_PATH),
        help="Path to processed CSV file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(ARTIFACTS_DIR),
        help="Base directory for output artifacts",
    )
    args = parser.parse_args()

    main(Path(args.data), Path(args.output))