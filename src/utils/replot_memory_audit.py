from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def iqr_bounds(series: pd.Series, multiplier: float = 1.5) -> tuple[float, float]:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return lower, upper


def remove_outliers_iqr(
    df: pd.DataFrame,
    columns: list[str],
    multiplier: float = 1.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Removes rows that are outliers in ANY of the specified columns using the IQR rule.
    Returns:
        filtered_df, removed_df
    """
    mask = pd.Series(True, index=df.index)

    for col in columns:
        valid = df[col].dropna()
        if valid.empty:
            continue

        lower, upper = iqr_bounds(valid, multiplier=multiplier)
        col_mask = df[col].between(lower, upper, inclusive="both") | df[col].isna()
        mask &= col_mask

    filtered_df = df[mask].copy()
    removed_df = df[~mask].copy()
    return filtered_df, removed_df


def make_scatter_with_legend(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    outpath: Path,
    title: str,
):
    plot_df = df.dropna(subset=[xcol, ycol]).copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Use matplotlib's default categorical palette
    cmap = plt.get_cmap("tab20")
    n = len(plot_df)

    for i, (_, row) in enumerate(plot_df.iterrows()):
        ax.scatter(
            row[xcol],
            row[ycol],
            label=row["website"],
            color=cmap(i % 20),
            s=60,
        )

    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(title)

    # Put legend outside the plot to avoid overlap
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
        fontsize=8,
        title="Website",
    )

    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Recreate memory-audit scatter plots with major outliers removed."
    )
    ap.add_argument(
        "--csv",
        default="results/website_memory_audit/website_feature_runtime_table.csv",
        help="Path to website_feature_runtime_table.csv",
    )
    ap.add_argument(
        "--outdir",
        default="results/website_memory_audit/no_outliers",
        help="Directory to save filtered CSV and plots",
    )
    ap.add_argument(
        "--multiplier",
        type=float,
        default=1.5,
        help="IQR multiplier for outlier detection (default: 1.5)",
    )
    ap.add_argument(
        "--filter_cols",
        nargs="+",
        default=["axtree_chars", "prompt_tokens"],
        help="Columns to use when detecting outliers",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    required_cols = {"website", "axtree_chars", "prompt_tokens", "webnav_peak_allocated_mb"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    filtered_df, removed_df = remove_outliers_iqr(
        df,
        columns=args.filter_cols,
        multiplier=args.multiplier,
    )

    filtered_csv = outdir / "website_feature_runtime_table_no_outliers.csv"
    removed_csv = outdir / "removed_outliers.csv"

    filtered_df.to_csv(filtered_csv, index=False)
    removed_df.to_csv(removed_csv, index=False)

    make_scatter_with_legend(
        df=filtered_df,
        xcol="axtree_chars",
        ycol="webnav_peak_allocated_mb",
        outpath=outdir / "scatter_axtree_chars_vs_webnav_vram_no_outliers.png",
        title="AXTree chars vs peak webnav VRAM (outliers removed)",
    )

    make_scatter_with_legend(
        df=filtered_df,
        xcol="prompt_tokens",
        ycol="webnav_peak_allocated_mb",
        outpath=outdir / "scatter_prompt_tokens_vs_webnav_vram_no_outliers.png",
        title="Prompt tokens vs peak webnav VRAM (outliers removed)",
    )

    print("\nSaved:")
    print(f"  Filtered CSV: {filtered_csv}")
    print(f"  Removed outliers CSV: {removed_csv}")
    print(f"  Plot: {outdir / 'scatter_axtree_chars_vs_webnav_vram_no_outliers.png'}")
    print(f"  Plot: {outdir / 'scatter_prompt_tokens_vs_webnav_vram_no_outliers.png'}")

    if not removed_df.empty:
        print("\nRemoved outlier websites:")
        for name in removed_df["website"].tolist():
            print(f"  - {name}")
    else:
        print("\nNo outliers were removed.")


if __name__ == "__main__":
    main()