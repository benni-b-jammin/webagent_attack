#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run_command(cmd: list[str], log_file, dry_run: bool = False) -> int:
    line = "[RUN] " + " ".join(cmd)
    print(line, flush=True)
    log_file.write(line + "\n")
    log_file.flush()

    if dry_run:
        return 0

    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")
    return result.returncode


def discover_trigger_configs(trigger_config_dir: Path, skip_default: bool = True) -> list[Path]:
    files = []
    for ext in ("*.yaml", "*.yml", "*.json"):
        files.extend(trigger_config_dir.glob(ext))

    files = sorted(files)

    if skip_default:
        files = [f for f in files if "default" not in f.name.lower()]

    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate trigger artifacts for all trigger configs in a directory."
    )
    parser.add_argument(
        "--trigger_config_dir",
        type=str,
        default="src/config/narrow_triggers",
        help="Directory containing trigger config files.",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="gcg",
        help="Trigger algorithm to pass to make_trigger.py",
    )
    parser.add_argument(
        "--n_triggers",
        type=int,
        default=1,
        help="Number of triggers to generate per config file.",
    )
    parser.add_argument(
        "--skip_default",
        action="store_true",
        help="Skip config files whose names contain 'default'.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing them.",
    )

    args = parser.parse_args()

    trigger_config_dir = (ROOT / args.trigger_config_dir).resolve()
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = results_dir / f"generate_triggers_log_{timestamp}.txt"

    with open(log_path, "w", encoding="utf-8") as log_file:
        header = f"Trigger generation started at {timestamp}\n"
        print(header.strip(), flush=True)
        log_file.write(header)

        if not trigger_config_dir.exists():
            raise FileNotFoundError(f"Trigger config directory does not exist: {trigger_config_dir}")

        files = discover_trigger_configs(
            trigger_config_dir=trigger_config_dir,
            skip_default=args.skip_default,
        )

        print(f"Found {len(files)} config files", flush=True)
        log_file.write(f"Found {len(files)} config files\n")

        for file in files:
            print(f"[CONFIG] {file}", flush=True)
            log_file.write(f"[CONFIG] {file}\n")

        for file in files:
            for i in range(args.n_triggers):
                print(
                    f"Using config: {file} (trigger {i + 1}/{args.n_triggers})",
                    flush=True,
                )
                log_file.write(
                    f"Using config: {file} (trigger {i + 1}/{args.n_triggers})\n"
                )
                log_file.flush()

                cmd = [
                    sys.executable,
                    "-m",
                    "src.attacks.make_trigger",
                    "--algo",
                    args.algo,
                    "--config",
                    str(file),
                ]
                run_command(cmd, log_file=log_file, dry_run=args.dry_run)

        print("Done.", flush=True)
        log_file.write("Done.\n")


if __name__ == "__main__":
    main()