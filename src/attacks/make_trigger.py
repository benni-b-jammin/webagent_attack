#!/usr/bin/env python3
"""
make_trigger.py (entrypoint)

Purpose
-------
Create a trigger artifact file. For now (per current project state), this script only supports:
  - fixed: writes a trigger string from config into data/triggers/<name>.json

This is a placeholder entrypoint so the rest of the pipeline (capture -> run_demo -> eval) is usable.
Trigger optimization algorithms (GCG, blackbox, etc.) will be added later.

Configuration
-------------
Defaults to src/config/trigger_default.yaml (if present). You can override with --config.

Output
------
Writes a JSON artifact with:
  - trigger string
  - metadata (algo, timestamp, config used)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import json
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.utils.config import load_config  # noqa: E402
from src.utils.logging_utils import make_run_dir, setup_logger  # noqa: E402

from src.utils.trigger_registry import init_registry, get, available  # noqa: E402
from src.utils.trigger_artifacts import TriggerArtifact, save_trigger  # noqa: E402
from src.utils.eval_runner import load_dataset_items  # noqa: E402

DEFAULT_CONFIG = "src/config/narrow_triggers/trigger_default.yaml"


def main() -> None:
    ap = argparse.ArgumentParser(description="Make trigger artifact (registry-based).")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help=f"Config path. Default: {DEFAULT_CONFIG}")
    ap.add_argument("--algo", default="fixed", help="Trigger algorithm name (from registry).")
    ap.add_argument("--dataset", default="data/datasets/demo_dataset", help="Dataset dir (optional for some algos).")
    ap.add_argument("--limit_items", type=int, default=None, help="Limit dataset items loaded.")
    ap.add_argument("--out", default="src/data/triggers/demo_trigger.json", help="Output trigger JSON path.")
    args = ap.parse_args()

    cfg = load_config(args.config, default={})

    run_dir = make_run_dir(prefix="make_trigger", base_dir="results")
    logger = setup_logger(run_dir=run_dir)

    # Initialize registry and resolve algorithm
    init_registry()
    if args.algo not in available():
        raise SystemExit(f"Unknown --algo {args.algo}. Available: {available()}")
    algo = get(args.algo)

    # Load dataset items if needed (safe default: empty list)
    items = []
    try:
        items = load_dataset_items(args.dataset, limit=args.limit_items)
    except Exception as e:
        logger.info(f"Dataset not loaded (may be OK for some algos). Reason: {e}")

    # Run algorithm (provider is None for fixed; future algos may require provider)
    result = algo.run(cfg=cfg, items=items, provider=None)
    trigger = result["trigger"]

    input_json_path = Path(cfg["json"])
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    output_base_dir = Path("src/data/triggers")
    out_path = output_base_dir / f"{input_json_path.stem}_{timestamp}.json"


    output = {
        "json_path": str(input_json_path),
        "model": cfg.get("model"),
        "target": cfg.get("target"),
        "trigger": trigger,
        "time_to_find": None if result.get("trace") is None else result["trace"].get("time_to_find"),
        "num_steps": None if result.get("trace") is None else result["trace"].get("num_steps"),
        "args": cfg,
    }

    logger.info(f"Writing trigger artifact to {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=4), encoding="utf-8")
    logger.info("Done.")


if __name__ == "__main__":
    main()