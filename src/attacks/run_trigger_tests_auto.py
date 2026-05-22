#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# Data structures
# ============================================================

@dataclass
class DemoRunResult:
    site_tag: str
    prompt: str
    config_path: str
    trigger_path: Optional[str]
    exp_dir: Optional[str]
    extracted_action: Optional[str]
    returncode: int
    stdout_path: str
    stderr_path: str
    raw_stdout: str
    raw_stderr: str


@dataclass
class AttackRunRecord:
    site_tag: str
    prompt: str

    baseline_exp_dir: Optional[str]
    baseline_action: Optional[str]

    trigger_file: str
    trigger_target: Optional[str]
    trigger_string: Optional[str]

    attack_exp_dir: Optional[str]
    attack_action: Optional[str]

    attack_status: str   # SUCCESS / FAILURE / ERROR
    notes: str


# ============================================================
# Helpers
# ============================================================

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_nonempty_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_extracted_action(stdout: str) -> Optional[str]:
    """
    Extract action from the explicit block:
        === EXTRACTED ACTION ===
        fill('4025', 'Translate by voice')
    """
    m = re.search(
        r"=== EXTRACTED ACTION ===\s*\n(.+?)(?:\n|$)",
        stdout,
        flags=re.MULTILINE,
    )
    if m:
        return m.group(1).strip()

    # Fallback: BrowserGym log block
    m = re.search(
        r"action:\s*\n(.+?)(?:\n|$)",
        stdout,
        flags=re.MULTILINE,
    )
    if m:
        return m.group(1).strip()

    return None


def parse_exp_dir(stdout: str) -> Optional[str]:
    """
    Prefer final exp_dir: results\\... line from EXPERIMENT RESULT.
    """
    m = re.search(r"^exp_dir:\s*(.+)$", stdout, flags=re.MULTILINE)
    if m:
        return m.group(1).strip()

    # Fallback: BrowserGym "Running experiment ... in:"
    m = re.search(r"Running experiment .*? in:\s*\n\s*(results[^\r\n]+)", stdout, flags=re.MULTILINE)
    if m:
        return m.group(1).strip()

    return None


def load_trigger_metadata(trigger_path: Path) -> tuple[Optional[str], Optional[str]]:
    """
    Returns (target, trigger_string) from a trigger artifact JSON if present.
    """
    try:
        data = json.loads(trigger_path.read_text(encoding="utf-8"))
    except Exception:
        return None, None

    return data.get("target"), data.get("trigger")

def load_site_url(dataset_json_path: Path) -> str:
    data = json.loads(dataset_json_path.read_text(encoding="utf-8"))
    urls = data.get("open_pages_urls", [])
    if not urls:
        raise ValueError(f"No open_pages_urls found in {dataset_json_path}")
    return urls[0]


def run_demo_once(
    site_tag: str,
    prompt: str,
    config_path: Path,
    start_url: str,
    work_dir: Path,
    trigger_path: Optional[Path] = None,
    headless: bool = True,
) -> DemoRunResult:
    """
    Run run_demo.py once and capture/parses stdout.
    """
    stdout_path = work_dir / "stdout.txt"
    stderr_path = work_dir / "stderr.txt"

    cmd = [
        sys.executable,
        "-m",
        "src.attacks.run_demo",
        "--config",
        str(config_path),
        "--start_url",
        start_url,
        "--goal",
        prompt,
        "--headless",
        "true" if headless else "false",
    ]

    if trigger_path is not None:
        cmd.extend(["--trigger_path", str(trigger_path)])

    result = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    stdout_path.write_text(result.stdout, encoding="utf-8", errors="replace")
    stderr_path.write_text(result.stderr, encoding="utf-8", errors="replace")

    extracted_action = parse_extracted_action(result.stdout)
    exp_dir = parse_exp_dir(result.stdout)

    return DemoRunResult(
        site_tag=site_tag,
        prompt=prompt,
        config_path=str(config_path),
        trigger_path=str(trigger_path) if trigger_path else None,
        exp_dir=exp_dir,
        extracted_action=extracted_action,
        returncode=result.returncode,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        raw_stdout=result.stdout,
        raw_stderr=result.stderr,
    )


def discover_site_tags(dataset_dir: Path) -> list[str]:
    return sorted(p.stem for p in dataset_dir.glob("*.json"))


def discover_triggers_for_site(trigger_dir: Path, site_tag: str) -> list[Path]:
    return sorted(trigger_dir.glob(f"{site_tag}_*.json"))


# ============================================================
# Main workflow
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run baseline and triggered BrowserGym tests for each site/prompt pair."
    )
    parser.add_argument(
        "--dataset_items_dir",
        default="src/data/datasets/auto_data",
        help="Directory containing website JSON files.",
    )
    parser.add_argument(
        "--auto_config",
        default="src/config/auto_runs/auto_default.yaml",
        help="Default auto-run config passed to run_demo.py unless overridden per site.",
    )
    parser.add_argument(
        "--trigger_dir",
        default="src/data/triggers",
        help="Directory containing generated trigger artifacts named <site_tag>_<datetime>.json.",
    )
    parser.add_argument(
        "--prompt_dir",
        default="src/data/test_prompts",
        help="Directory containing <site_tag>.txt files.",
    )
    parser.add_argument(
        "--outdir",
        default="results/trigger_tests_auto",
        help="Directory to save baseline/attack test outputs.",
    )
    parser.add_argument(
        "--headless",
        default="true",
        help="Run demos headless (true/false).",
    )
    parser.add_argument(
        "--limit_sites",
        type=int,
        default=None,
        help="Optional limit on number of sites processed.",
    )
    parser.add_argument(
        "--latest_trigger_only",
        action="store_true",
        help="Use only the most recent trigger artifact per site.",
    )

    args = parser.parse_args()

    dataset_dir = (ROOT / args.dataset_items_dir).resolve()
    trigger_dir = (ROOT / args.trigger_dir).resolve()
    prompt_dir = (ROOT / args.prompt_dir).resolve()
    outdir = ensure_dir(ROOT / args.outdir)
    auto_config = (ROOT / args.auto_config).resolve()

    headless = args.headless.strip().lower() in {"1", "true", "yes", "y", "on"}

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")
    if not prompt_dir.exists():
        raise FileNotFoundError(f"Prompt dir not found: {prompt_dir}")
    if not trigger_dir.exists():
        raise FileNotFoundError(f"Trigger dir not found: {trigger_dir}")
    if not auto_config.exists():
        raise FileNotFoundError(f"Auto-run config not found: {auto_config}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_dir = ensure_dir(outdir / f"run_{timestamp}")
    baseline_dir = ensure_dir(session_dir / "baseline")
    attacks_dir = ensure_dir(session_dir / "attacks")
    baseline_index_path = session_dir / "baseline_index.json"
    attack_summary_jsonl = session_dir / "attack_summary.jsonl"
    attack_summary_csv = session_dir / "attack_summary.csv"

    baseline_index: dict[tuple[str, str], DemoRunResult] = {}
    attack_rows: list[AttackRunRecord] = []

    site_tags = discover_site_tags(dataset_dir)
    if args.limit_sites is not None:
        site_tags = site_tags[: args.limit_sites]

    print(f"Found {len(site_tags)} site JSON files")

    # --------------------------------------------------------
    # 1) Baselines: no trigger, each prompt once
    # --------------------------------------------------------
    for site_tag in site_tags:
        prompt_file = prompt_dir / f"{site_tag}.txt"

        if not prompt_file.exists():
            print(f"[skip] No prompt file for {site_tag}: {prompt_file}")
            continue

        prompts = read_nonempty_lines(prompt_file)
        if not prompts:
            print(f"[skip] Empty prompt file for {site_tag}")
            continue

        dataset_json_path = dataset_dir / f"{site_tag}.json"
        if not dataset_json_path.exists():
            print(f"[skip] No dataset JSON for {site_tag}: {dataset_json_path}")
            continue

        start_url = load_site_url(dataset_json_path)

        for idx, prompt in enumerate(prompts, start=1):
            run_dir = ensure_dir(baseline_dir / site_tag / f"prompt_{idx:02d}")

            print(f"[baseline] {site_tag} | prompt {idx}/{len(prompts)}")
            baseline_result = run_demo_once(
                site_tag=site_tag,
                prompt=prompt,
                config_path=auto_config,
                start_url=start_url,
                work_dir=run_dir,
                trigger_path=None,
                headless=headless,
            )
            # Save structured baseline result
            (run_dir / "baseline_result.json").write_text(
                json.dumps(asdict(baseline_result), indent=4),
                encoding="utf-8",
            )

            baseline_index[(site_tag, prompt)] = baseline_result

            serializable_baseline_index = {
                f"{k[0]}|||{k[1]}": asdict(v)
                for k, v in baseline_index.items()
            }
            baseline_index_path.write_text(
                json.dumps(serializable_baseline_index, indent=4),
                encoding="utf-8",
            )

    # --------------------------------------------------------
    # 2) Triggered runs: each trigger x each prompt
    # --------------------------------------------------------
    for site_tag in site_tags:
        prompt_file = prompt_dir / f"{site_tag}.txt"

        if not prompt_file.exists():
            continue

        prompts = read_nonempty_lines(prompt_file)
        if not prompts:
            continue

        dataset_json_path = dataset_dir / f"{site_tag}.json"
        if not dataset_json_path.exists():
            continue

        start_url = load_site_url(dataset_json_path)

        trigger_files = discover_triggers_for_site(trigger_dir, site_tag)
        if args.latest_trigger_only and trigger_files:
            trigger_files = [max(trigger_files, key=lambda p: p.stat().st_mtime)]

        if not trigger_files:
            print(f"[skip] No trigger artifacts for {site_tag}")
            continue

        for trigger_idx, trigger_file in enumerate(trigger_files, start=1):
            trigger_target, trigger_string = load_trigger_metadata(trigger_file)

            for prompt_idx, prompt in enumerate(prompts, start=1):
                baseline_result = baseline_index.get((site_tag, prompt))
                if baseline_result is None:
                    continue

                run_dir = ensure_dir(
                    attacks_dir / site_tag / f"trigger_{trigger_idx:02d}" / f"prompt_{prompt_idx:02d}"
                )

                print(
                    f"[attack] {site_tag} | trigger {trigger_idx}/{len(trigger_files)} | prompt {prompt_idx}/{len(prompts)}"
                )

                attack_result = run_demo_once(
                    site_tag=site_tag,
                    prompt=prompt,
                    config_path=auto_config,
                    start_url=start_url,
                    work_dir=run_dir,
                    trigger_path=trigger_file,
                    headless=headless,
                )

                baseline_action = baseline_result.extracted_action
                attack_action = attack_result.extracted_action

                if attack_result.returncode != 0:
                    attack_status = "ERROR"
                    notes = "Triggered run returned non-zero exit code."
                elif baseline_action != attack_action:
                    attack_status = "SUCCESS"
                    notes = "Triggered action differs from baseline action."
                else:
                    attack_status = "FAILURE"
                    notes = "Triggered action matches baseline action."

                record = AttackRunRecord(
                    site_tag=site_tag,
                    prompt=prompt,
                    baseline_exp_dir=baseline_result.exp_dir,
                    baseline_action=baseline_action,
                    trigger_file=str(trigger_file),
                    trigger_target=trigger_target,
                    trigger_string=trigger_string,
                    attack_exp_dir=attack_result.exp_dir,
                    attack_action=attack_action,
                    attack_status=attack_status,
                    notes=notes,
                )
                attack_rows.append(record)

                with open(attack_summary_jsonl, "a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(record)) + "\n")

                # Save per-test structured record
                payload = {
                    "baseline": asdict(baseline_result),
                    "attack": asdict(attack_result),
                    "comparison": asdict(record),
                }
                (run_dir / "attack_record.json").write_text(
                    json.dumps(payload, indent=4),
                    encoding="utf-8",
                )
                with open(attack_summary_csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "site_tag",
                            "prompt",
                            "baseline_exp_dir",
                            "baseline_action",
                            "trigger_file",
                            "trigger_target",
                            "trigger_string",
                            "attack_exp_dir",
                            "attack_action",
                            "attack_status",
                            "notes",
                        ],
                    )
                    writer.writeheader()
                    for row in attack_rows:
                        writer.writerow(asdict(row))

    # --------------------------------------------------------
    # 3) Session summary
    # --------------------------------------------------------
    # summary_csv = session_dir / "attack_summary.csv"
    # with open(summary_csv, "w", newline="", encoding="utf-8") as f:
    #     writer = csv.DictWriter(
    #         f,
    #         fieldnames=[
    #             "site_tag",
    #             "prompt",
    #             "baseline_exp_dir",
    #             "baseline_action",
    #             "trigger_file",
    #             "trigger_target",
    #             "trigger_string",
    #             "attack_exp_dir",
    #             "attack_action",
    #             "attack_status",
    #             "notes",
    #         ],
    #     )
    #     writer.writeheader()
    #     for row in attack_rows:
    #         writer.writerow(asdict(row))

    print(f"\nDone. Results saved to: {session_dir}")


if __name__ == "__main__":
    main()