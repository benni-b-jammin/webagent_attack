import argparse
import subprocess
import sys
from pathlib import Path

import src.config.auto_config as auto_cfg


# ============================================================
# Helpers
# ============================================================

def run_command(cmd: list[str], dry_run: bool = False) -> int:
    """
    Print and run a subprocess command.
    """
    print("\n[RUN]", " ".join(cmd), flush=True)
    if dry_run:
        return 0

    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")
    return result.returncode


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


# ============================================================
# Step 1: Retrieve and capture websites
# ============================================================

def step1_get_websites(
    query_types_file: str,
    n_websites: int,
    n_search_queries: int,
    urls_per_query: int,
    captured_sites_dir: str,
    dry_run: bool = False,
) -> None:
    """
    Calls the implemented dataset retrieval/capture stage.
    """
    ensure_dir(captured_sites_dir)

    cmd = [
        sys.executable,
        "-m",
        "src.attacks.capture_data_automated",
        "get_webs",
        "--query_types_file", query_types_file,
        "--n_websites", str(n_websites),
        "--n_search_queries", str(n_search_queries),
        "--urls_per_query", str(urls_per_query),
        "--out_dir", captured_sites_dir,
    ]
    run_command(cmd, dry_run=dry_run)


# ============================================================
# Step 2: Audit websites and remove infeasible ones
# ============================================================

def step2_memory_audit(
    captured_sites_dir: str,
    audit_outdir: str,
    model: str,
    trigger_config_dir: str,
    default_trigger_name: str,
    measure_trigger: bool,
    trigger_audit_steps: int,
    delete_failed_webnav: bool,
    delete_failed_trigger: bool,
    dry_run: bool = False,
) -> None:
    """
    Calls the implemented memory-audit stage.
    """
    ensure_dir(audit_outdir)

    cmd = [
        sys.executable,
        "src/analysis/web_memory_audit.py",
        "--dataset_items_dir", captured_sites_dir,
        "--outdir", audit_outdir,
        "--model", model,
        "--trigger_config_dir", trigger_config_dir,
        "--default_trigger_name", default_trigger_name,
        "--trigger_audit_steps", str(trigger_audit_steps),
    ]

    if measure_trigger:
        cmd.append("--measure_trigger")
    if delete_failed_webnav:
        cmd.append("--delete_failed_webnav")
    if delete_failed_trigger:
        cmd.append("--delete_failed_trigger")

    run_command(cmd, dry_run=dry_run)


# ============================================================
# Step 3: Determine trigger action + 10 regular actions
# ============================================================

def step3_generate_actions(
            captured_sites_dir: str,
            prompts_dir: str,
            meta_dir: str,
            trigger_config_dir: str,
            summary_csv: str,
            model: str,
            n_regular_actions: int,
            skip_existing: bool,
            dry_run: bool = False,
    ) -> None:
        """
        Generate one trigger action, 10 regular actions, and per-site trigger YAML files
        for each surviving website JSON after memory audit.
        """
        ensure_dir(prompts_dir)
        ensure_dir(meta_dir)
        ensure_dir(trigger_config_dir)
        ensure_dir(Path(summary_csv).parent)

        cmd = [
            sys.executable,
            "-m",
            "src.attacks.generate_site_actions",
            "--dataset_items_dir", captured_sites_dir,
            "--prompts_dir", prompts_dir,
            "--meta_dir", meta_dir,
            "--trigger_config_dir", trigger_config_dir,
            "--summary_csv", summary_csv,
            "--model", model,
            "--n_regular_actions", str(n_regular_actions),
        ]

        if skip_existing:
            cmd.append("--skip_existing")

        run_command(cmd, dry_run=dry_run)


# ============================================================
# Step 4: Generate triggers
# ============================================================

def step4_generate_triggers(
    trigger_config_dir: str,
    n_triggers_per_site: int,
    trigger_algo: str,
    skip_default_trigger_config: bool,
    dry_run: bool = False,
) -> None:
    """
    Generate trigger artifacts for each per-site trigger config.
    """
    ensure_dir(trigger_config_dir)

    cmd = [
        sys.executable,
        "-m",
        "src.attacks.generate_triggers_auto",
        "--trigger_config_dir", trigger_config_dir,
        "--algo", trigger_algo,
        "--n_triggers", str(n_triggers_per_site),
    ]

    if skip_default_trigger_config:
        cmd.append("--skip_default")

    if dry_run:
        cmd.append("--dry_run")
    run_command(cmd, dry_run=False)


# ============================================================
# Step 5: Run baselines and attacked demos
# ============================================================

def step5_run_tests(
    captured_sites_dir: str,
    prompts_dir: str,
    trigger_dir: str,
    auto_config: str,
    results_dir: str,
    headless: bool = True,
    latest_trigger_only: bool = False,
    limit_sites: int | None = None,
    dry_run: bool = False,
) -> None:
    """
    Run baseline and triggered BrowserGym tests using run_demo.py through the
    Step 5 wrapper script.
    """
    ensure_dir(results_dir)

    cmd = [
        sys.executable,
        "-m",
        "src.attacks.run_trigger_tests_auto",
        "--dataset_items_dir", captured_sites_dir,
        "--prompt_dir", prompts_dir,
        "--trigger_dir", trigger_dir,
        "--auto_config", auto_config,
        "--outdir", results_dir,
        "--headless", "true" if headless else "false",
    ]

    if latest_trigger_only:
        cmd.append("--latest_trigger_only")

    if limit_sites is not None:
        cmd.extend(["--limit_sites", str(limit_sites)])

    run_command(cmd, dry_run=dry_run)


# ============================================================
# Step 6: Summarize final results
# ============================================================

def step6_summarize_results(
    captured_sites_dir: str,
    results_dir: str,
    summary_outdir: str,
    attack_summary_csv: str | None,
    memory_summary_csv: str,
    memory_feature_csv: str,
    dry_run: bool = False,
) -> None:
    """
    Run Step 6 summarization from Step 5 outputs and Step 2 memory-audit outputs.
    If attack_summary_csv is not provided, use the newest run in results_dir.
    """
    ensure_dir(summary_outdir)

    resolved_attack_summary = attack_summary_csv

    if not resolved_attack_summary:
        results_root = Path(results_dir)
        candidate_runs = sorted(
            [p for p in results_root.glob("run_*") if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if not candidate_runs:
            raise FileNotFoundError(
                f"No Step 5 run directories found in {results_root}"
            )

        latest_run = candidate_runs[0]
        candidate_csv = latest_run / "attack_summary.csv"
        if not candidate_csv.exists():
            raise FileNotFoundError(
                f"No attack_summary.csv found in latest Step 5 run directory: {latest_run}"
            )

        resolved_attack_summary = str(candidate_csv)

    cmd = [
        sys.executable,
        "-m",
        "src.attacks.summarize_trigger_tests",
        "--attack_summary_csv", resolved_attack_summary,
        "--dataset_items_dir", captured_sites_dir,
        "--memory_summary_csv", memory_summary_csv,
        "--memory_feature_csv", memory_feature_csv,
        "--outdir", summary_outdir,
    ]

    run_command(cmd, dry_run=dry_run)


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the end-to-end automated web-agent attack pipeline."
    )

    # ----------------------------
    # General
    # ----------------------------
    parser.add_argument("--dry_run", action="store_true", help="Print commands but do not execute them.")
    parser.add_argument(
        "--start_step",
        type=int,
        default=auto_cfg.START_STEP,
        choices=[1, 2, 3, 4, 5, 6],
        help="Pipeline step to start from.",
    )
    parser.add_argument(
        "--end_step",
        type=int,
        default=auto_cfg.END_STEP,
        choices=[1, 2, 3, 4, 5, 6],
        help="Pipeline step to stop after.",
    )

    # ----------------------------
    # Step 1
    # ----------------------------
    parser.add_argument("--query_types_file", type=str, default=auto_cfg.QUERY_TYPES_FILE,
                        help="Input text file of webpage types.")
    parser.add_argument("--n_websites", type=int, default=auto_cfg.N_WEBSITES, help="Number of websites to capture.")
    parser.add_argument("--n_search_queries", type=int, default=auto_cfg.N_SEARCH_QUERIES,
                        help="Number of OpenAI-generated search queries.")
    parser.add_argument("--urls_per_query", type=int, default=auto_cfg.URLS_PER_QUERY,
                        help="Number of search-result URLs per query.")
    parser.add_argument("--captured_sites_dir", type=str, default=auto_cfg.CAPTURED_SITES_DIR,
                        help="Directory for captured site JSONs.")

    # ----------------------------
    # Step 2
    # ----------------------------
    parser.add_argument("--audit_outdir", type=str, default=auto_cfg.AUDIT_OUTDIR,
                        help="Directory for memory-audit outputs.")
    parser.add_argument("--model", type=str, default=auto_cfg.MODEL, help="Model used for memory auditing.")
    parser.add_argument("--trigger_config_dir", type=str, default=auto_cfg.TRIGGER_CONFIG_DIR,
                        help="Directory containing trigger YAML files & where generated YAML files are saved.")
    parser.add_argument("--default_trigger_name", type=str, default=auto_cfg.DEFAULT_TRIGGER_NAME,
                        help="Fallback trigger YAML file.")
    parser.add_argument("--measure_trigger", action="store_true", help="Also perform trigger-feasibility audit.")
    parser.add_argument("--trigger_audit_steps", type=int, default=auto_cfg.TRIGGER_AUDIT_STEPS,
                        help="Number of trigger steps used in audit mode.")
    parser.add_argument("--delete_failed_webnav", action="store_true",
                        help="Delete sites that fail webnav memory audit.")
    parser.add_argument("--delete_failed_trigger", action="store_true", help="Delete sites that fail trigger audit.")

    # ----------------------------
    # Step 3
    # ----------------------------
    parser.add_argument("--prompts_dir", type=str, default="src/data/test_prompts",
                        help="Directory to save and access generated regular-action prompt files.")
    parser.add_argument("--meta_dir", type=str, default="src/data/task_meta",
                        help="Directory to save generated per-site action metadata JSON files.")
    parser.add_argument("--actions_summary_csv", type=str, default="results/generated_site_actions_summary.csv",
                        help="Summary CSV for generated site actions.")
    parser.add_argument("--action_model", type=str, default="gpt-4o-mini",
                        help="OpenAI model to use for generating trigger actions and regular prompts.")
    parser.add_argument("--n_regular_actions", type=int, default=10,
                        help="Number of regular test prompts to generate per site.")
    parser.add_argument("--skip_existing_actions", action="store_true",
                        help="Skip sites whose generated action metadata already exists.")

    # ----------------------------
    # Step 4
    # ----------------------------
    parser.add_argument("--n_triggers_per_site", type=int, default=1,
                        help="Number of triggers to generate per surviving website.")
    parser.add_argument("--trigger_algo", type=str, default="gcg",
                        help="Trigger-generation algorithm to pass to make_trigger.py.")
    parser.add_argument("--skip_default_trigger_config", action="store_true",
                        help="Skip config files whose names contain 'default' during trigger generation.")
    # ----------------------------
    # Step 5
    # ----------------------------
    parser.add_argument("--trigger_dir", type=str, default=auto_cfg.TRIGGER_DIR,
                        help="Directory containing generated trigger artifacts.")
    parser.add_argument("--auto_config", type=str, default=auto_cfg.AUTO_RUN_CONFIG,
                        help="Default auto-run config passed to run_demo.py and overridden by CLI arguments.")
    parser.add_argument("--headless_tests", type=str, default=auto_cfg.HEADLESS_TESTS,
                        help="Run Step 5 demos in headless mode (true/false).")
    parser.add_argument("--latest_trigger_only", action="store_true",
                        help="Use only the most recent trigger artifact per site in Step 5.")
    parser.add_argument("--limit_test_sites", type=int, default=None,
                        help="Optional limit on the number of Step 5 sites processed.")
    parser.add_argument("--results_dir", type=str, default=auto_cfg.RESULTS_DIR,
                        help="Directory for Step 5 run outputs.")

    # ----------------------------
    # Step 6
    # ----------------------------
    parser.add_argument("--summary_outdir", type=str, default=auto_cfg.SUMMARY_OUTDIR, help="Directory for final summaries and graphs.")
    parser.add_argument("--attack_summary_csv", type=str, default=auto_cfg.ATTACK_SUMMARY_CSV, help="Optional explicit path to Step 5 attack_summary.csv. If omitted, auto-detect from results_dir.")
    parser.add_argument("--memory_summary_csv", type=str, default=auto_cfg.MEMORY_SUMMARY_CSV, help="Path to Step 2 website_summary_table.csv")
    parser.add_argument("--memory_feature_csv", type=str, default=auto_cfg.MEMORY_FEATURE_CSV, help="Path to Step 2 website_feature_runtime_table.csv")

    args = parser.parse_args()

    if args.start_step > args.end_step:
        raise ValueError("--start_step must be <= --end_step")

    # Step 1
    if args.start_step <= 1 <= args.end_step:
        step1_get_websites(
            query_types_file=args.query_types_file,
            n_websites=args.n_websites,
            n_search_queries=args.n_search_queries,
            urls_per_query=args.urls_per_query,
            captured_sites_dir=args.captured_sites_dir,
            dry_run=args.dry_run,
        )

    # Step 2
    if args.start_step <= 2 <= args.end_step:
        step2_memory_audit(
            captured_sites_dir=args.captured_sites_dir,
            audit_outdir=args.audit_outdir,
            model=args.model,
            trigger_config_dir=args.trigger_config_dir,
            default_trigger_name=args.default_trigger_name,
            measure_trigger=args.measure_trigger,
            trigger_audit_steps=args.trigger_audit_steps,
            delete_failed_webnav=args.delete_failed_webnav,
            delete_failed_trigger=args.delete_failed_trigger,
            dry_run=args.dry_run,
        )

    # Step 3
    if args.start_step <= 3 <= args.end_step:
        step3_generate_actions(
            captured_sites_dir=args.captured_sites_dir,
            prompts_dir=args.prompts_dir,
            meta_dir=args.meta_dir,
            trigger_config_dir=args.trigger_config_dir,
            summary_csv=args.actions_summary_csv,
            model=args.action_model,
            n_regular_actions=args.n_regular_actions,
            skip_existing=args.skip_existing_actions,
            dry_run=args.dry_run,
        )

    # Step 4
    if args.start_step <= 4 <= args.end_step:
        step4_generate_triggers(
            trigger_config_dir=args.trigger_config_dir,
            trigger_algo=args.trigger_algo,
            n_triggers_per_site=args.n_triggers_per_site,
            dry_run=args.dry_run,
            skip_default_trigger_config=args.skip_default_trigger_config,
        )

    # Step 5
    if args.start_step <= 5 <= args.end_step:
        step5_run_tests(
            captured_sites_dir=args.captured_sites_dir,
            prompts_dir=args.prompts_dir,
            trigger_dir=args.trigger_dir,
            auto_config=args.auto_config,
            results_dir=args.results_dir,
            headless=args.headless_tests.strip().lower() in {"1", "true", "yes", "y", "on"},
            latest_trigger_only=args.latest_trigger_only,
            limit_sites=args.limit_test_sites,
            dry_run=args.dry_run,
        )

    # Step 6
    if args.start_step <= 6 <= args.end_step:
        step6_summarize_results(
            captured_sites_dir=args.captured_sites_dir,
            results_dir=args.results_dir,
            summary_outdir=args.summary_outdir,
            attack_summary_csv=args.attack_summary_csv,
            memory_summary_csv=args.memory_summary_csv,
            memory_feature_csv=args.memory_feature_csv,
            dry_run=args.dry_run,
        )

if __name__ == "__main__":
    main()