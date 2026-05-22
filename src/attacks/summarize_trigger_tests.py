from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_source_types(dataset_items_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(dataset_items_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        site_tag = data.get("site_tag") or path.stem
        source_type = data.get("source_type", "")
        source_query = data.get("source_query", "")

        rows.append({
            "site_tag": site_tag,
            "source_type": source_type,
            "source_query": source_query,
            "dataset_json": str(path),
        })

    return pd.DataFrame(rows)


def make_scatter(df: pd.DataFrame, xcol: str, ycol: str, outpath: Path, title: str):
    plot_df = df.dropna(subset=[xcol, ycol]).copy()
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(plot_df[xcol], plot_df[ycol])

    for _, row in plot_df.iterrows():
        ax.annotate(str(row["site_tag"]), (row[xcol], row[ycol]), fontsize=8)

    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize Step 5 trigger-test results and generate tables/plots."
    )
    parser.add_argument(
        "--attack_summary_csv",
        required=True,
        help="Path to Step 5 attack_summary.csv",
    )
    parser.add_argument(
        "--dataset_items_dir",
        required=True,
        help="Directory containing website JSON files with source_type.",
    )
    parser.add_argument(
        "--memory_summary_csv",
        default="src/results/website_memory_audit/website_summary_table.csv",
        help="Path to Step 2 website_summary_table.csv",
    )
    parser.add_argument(
        "--memory_feature_csv",
        default="src/results/website_memory_audit/website_feature_runtime_table.csv",
        help="Path to Step 2 website_feature_runtime_table.csv",
    )
    parser.add_argument(
        "--outdir",
        default="results/final_summary",
        help="Directory to save summary tables and plots.",
    )
    args = parser.parse_args()

    outdir = ensure_dir(args.outdir)

    # ---------------------------------------------------------
    # Load step 5 attack data
    # ---------------------------------------------------------
    attack_df = pd.read_csv(args.attack_summary_csv)
    attack_df["attack_status"] = attack_df["attack_status"].astype(str).str.upper()
    attack_df["is_success"] = attack_df["attack_status"].eq("SUCCESS")
    attack_df["is_failure"] = attack_df["attack_status"].eq("FAILURE")
    attack_df["is_error"] = attack_df["attack_status"].eq("ERROR")

    attack_df["trigger_name"] = attack_df["trigger_file"].fillna("").apply(
        lambda x: Path(x).stem if x else ""
    )

    # ---------------------------------------------------------
    # Load source types from dataset JSONs
    # ---------------------------------------------------------
    type_df = load_source_types(Path(args.dataset_items_dir))

    # ---------------------------------------------------------
    # Per-website overall success summary
    # ---------------------------------------------------------
    website_summary = (
        attack_df.groupby("site_tag", dropna=False)
        .agg(
            n_attempts=("site_tag", "size"),
            n_successes=("is_success", "sum"),
            n_failures=("is_failure", "sum"),
            n_errors=("is_error", "sum"),
            n_unique_triggers=("trigger_name", "nunique"),
            n_unique_prompts=("prompt", "nunique"),
        )
        .reset_index()
    )
    website_summary["success_rate"] = 100 * website_summary["n_successes"] / website_summary["n_attempts"]
    website_summary = website_summary.merge(type_df[["site_tag", "source_type"]], on="site_tag", how="left")
    website_summary.to_csv(outdir / "website_success_summary.csv", index=False)

    # ---------------------------------------------------------
    # Per-website, per-trigger summary
    # ---------------------------------------------------------
    trigger_by_site_summary = (
        attack_df.groupby(["site_tag", "trigger_name"], dropna=False)
        .agg(
            n_attempts=("site_tag", "size"),
            n_successes=("is_success", "sum"),
            n_failures=("is_failure", "sum"),
            n_errors=("is_error", "sum"),
            n_prompts=("prompt", "nunique"),
        )
        .reset_index()
    )
    trigger_by_site_summary["success_rate"] = (
        100 * trigger_by_site_summary["n_successes"] / trigger_by_site_summary["n_attempts"]
    )
    trigger_by_site_summary = trigger_by_site_summary.merge(
        type_df[["site_tag", "source_type"]],
        on="site_tag",
        how="left",
    )
    trigger_by_site_summary.to_csv(outdir / "trigger_success_by_website.csv", index=False)

    # ---------------------------------------------------------
    # Per-trigger overall summary
    # ---------------------------------------------------------
    trigger_summary = (
        attack_df.groupby("trigger_name", dropna=False)
        .agg(
            site_tag=("site_tag", "first"),
            n_attempts=("trigger_name", "size"),
            n_successes=("is_success", "sum"),
            n_failures=("is_failure", "sum"),
            n_errors=("is_error", "sum"),
        )
        .reset_index()
    )
    trigger_summary["success_rate"] = 100 * trigger_summary["n_successes"] / trigger_summary["n_attempts"]
    trigger_summary = trigger_summary.merge(
        type_df[["site_tag", "source_type"]],
        on="site_tag",
        how="left",
    )
    trigger_summary.to_csv(outdir / "trigger_success_summary.csv", index=False)

    # ---------------------------------------------------------
    # Per-type summary
    # ---------------------------------------------------------
    type_summary = (
        website_summary.groupby("source_type", dropna=False)
        .agg(
            n_websites=("site_tag", "nunique"),
            total_attempts=("n_attempts", "sum"),
            total_successes=("n_successes", "sum"),
            mean_website_success_rate=("success_rate", "mean"),
            min_website_success_rate=("success_rate", "min"),
            max_website_success_rate=("success_rate", "max"),
        )
        .reset_index()
    )
    type_summary["overall_type_success_rate"] = 100 * type_summary["total_successes"] / type_summary["total_attempts"]
    type_summary.to_csv(outdir / "website_type_success_summary.csv", index=False)

    # ---------------------------------------------------------
    # Overall totals
    # ---------------------------------------------------------
    overall = pd.DataFrame(
        [{
            "n_websites_with_results": attack_df["site_tag"].nunique(),
            "n_total_attempts": len(attack_df),
            "n_successes": int(attack_df["is_success"].sum()),
            "n_failures": int(attack_df["is_failure"].sum()),
            "n_errors": int(attack_df["is_error"].sum()),
            "overall_success_rate": 100 * attack_df["is_success"].sum() / len(attack_df) if len(attack_df) else 0.0,
        }]
    )
    overall.to_csv(outdir / "overall_results.csv", index=False)

    # ---------------------------------------------------------
    # Prompt-level detail
    # ---------------------------------------------------------
    prompt_detail = attack_df.merge(type_df[["site_tag", "source_type"]], on="site_tag", how="left")
    prompt_detail.to_csv(outdir / "prompt_level_results.csv", index=False)

    # ---------------------------------------------------------
    # Merge with memory audit outputs
    # ---------------------------------------------------------
    memory_summary_path = Path(args.memory_summary_csv)
    memory_feature_path = Path(args.memory_feature_csv)

    merged_memory = None
    if memory_summary_path.exists():
        mem_summary = pd.read_csv(memory_summary_path)
        mem_summary = mem_summary.rename(columns={"website": "site_tag"})
        merged_memory = website_summary.merge(mem_summary, on="site_tag", how="left")
    elif memory_feature_path.exists():
        mem_feature = pd.read_csv(memory_feature_path)
        mem_feature = mem_feature.rename(columns={"website": "site_tag"})
        merged_memory = website_summary.merge(mem_feature, on="site_tag", how="left")

    if merged_memory is not None:
        merged_memory.to_csv(outdir / "website_success_vs_memory.csv", index=False)

        # Correlations
        candidate_metrics = [
            "webnav_peak_allocated_mb",
            "trigger_peak_allocated_mb",
            "prompt_tokens",
            "axtree_chars",
            "axtree_tokens",
            "axtree_interactive_est",
        ]

        corr_rows = []
        for metric in candidate_metrics:
            if metric in merged_memory.columns:
                tmp = merged_memory[["success_rate", metric]].dropna()
                if len(tmp) >= 2:
                    corr_rows.append({
                        "metric": metric,
                        "pearson_corr": tmp["success_rate"].corr(tmp[metric], method="pearson"),
                        "spearman_corr": tmp["success_rate"].corr(tmp[metric], method="spearman"),
                        "n_sites": len(tmp),
                    })

        pd.DataFrame(corr_rows).to_csv(outdir / "correlation_summary.csv", index=False)

        # Scatter plots
        if "webnav_peak_allocated_mb" in merged_memory.columns:
            make_scatter(
                merged_memory,
                "webnav_peak_allocated_mb",
                "success_rate",
                outdir / "success_vs_webnav_vram.png",
                "Website Success Rate vs WebNav Peak VRAM",
            )

        if "trigger_peak_allocated_mb" in merged_memory.columns:
            make_scatter(
                merged_memory,
                "trigger_peak_allocated_mb",
                "success_rate",
                outdir / "success_vs_trigger_vram.png",
                "Website Success Rate vs Trigger Peak VRAM",
            )

        if "prompt_tokens" in merged_memory.columns:
            make_scatter(
                merged_memory,
                "prompt_tokens",
                "success_rate",
                outdir / "success_vs_prompt_tokens.png",
                "Website Success Rate vs Prompt Tokens",
            )

        if "axtree_chars" in merged_memory.columns:
            make_scatter(
                merged_memory,
                "axtree_chars",
                "success_rate",
                outdir / "success_vs_axtree_chars.png",
                "Website Success Rate vs AXTree Characters",
            )

    # ---------------------------------------------------------
    # Plot: per-website success rate
    # ---------------------------------------------------------
    website_plot_df = website_summary.sort_values("success_rate", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(website_plot_df["site_tag"], website_plot_df["success_rate"])
    ax.set_title("Attack Success Rate by Website")
    ax.set_xlabel("Website")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 100)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(outdir / "website_success_bar_chart.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------
    # Plot: per-trigger success by website
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    for site_tag, grp in trigger_by_site_summary.groupby("site_tag"):
        grp = grp.sort_values("trigger_name")
        ax.plot(grp["trigger_name"], grp["success_rate"], marker="o", linewidth=1.5, label=site_tag)

    ax.set_title("Trigger Success Rate by Website and Trigger")
    ax.set_xlabel("Trigger")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 100)
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(outdir / "trigger_success_line_chart.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------
    # Plot: outcome breakdown
    # ---------------------------------------------------------
    status_counts = pd.Series({
        "SUCCESS": int(attack_df["is_success"].sum()),
        "FAILURE": int(attack_df["is_failure"].sum()),
        "ERROR": int(attack_df["is_error"].sum()),
    })

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(status_counts.index, status_counts.values)
    ax.set_title("Attack Outcome Breakdown")
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(outdir / "attack_status_breakdown.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------
    # Plot: website type success rates
    # ---------------------------------------------------------
    type_plot_df = type_summary.sort_values("overall_type_success_rate", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(type_plot_df["source_type"], type_plot_df["overall_type_success_rate"])
    ax.set_title("Attack Success Rate by Website Type")
    ax.set_xlabel("Website Type")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 100)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(outdir / "website_type_success_bar_chart.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved summaries and plots to: {outdir}")


if __name__ == "__main__":
    main()