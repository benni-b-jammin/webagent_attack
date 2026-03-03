#!/usr/bin/env python3
"""
eval_trigger.py (entrypoint)

Purpose
-------
Evaluate a trigger (or placeholder "{optim_str}") over a dataset of captured pages in prompt-only mode.
For each dataset item:
  - build prompt with injected trigger
  - query the configured provider (OpenAI / HF local / HF hosted)
  - compute simple metrics (ACTION line present; optional substring match)

Outputs
-------
Writes a JSON report to results/eval_<timestamp>.json containing:
  - summary metrics
  - per-item model outputs

Configuration
-------------
Defaults to configs/demo_run.yaml for provider/model settings. Override with --config.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.config import load_config  # noqa: E402
from utils.providers import ProviderConfig, make_provider  # noqa: E402
from utils.eval_runner import eval_trigger_prompt_only  # noqa: E402
from utils.logging_utils import make_run_dir, setup_logger, write_json  # noqa: E402


DEFAULT_CONFIG = "configs/demo_run.yaml"


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate trigger over a dataset (prompt-only).")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help=f"Config path. Default: {DEFAULT_CONFIG}")
    ap.add_argument("--dataset", default="data/datasets/demo_dataset", help="Dataset directory.")
    ap.add_argument("--trigger", default=None, help="Optional trigger artifact JSON with `trigger` field.")
    ap.add_argument("--limit_items", type=int, default=None, help="Limit number of dataset items.")
    ap.add_argument("--target_contains", default=None, help="Optional substring to score for.")
    ap.add_argument("--out", default=None, help="Output report path (default in results/).")
    args = ap.parse_args()

    cfg = load_config(args.config)

    provider_name = cfg.get("provider", "openai")
    model = cfg.get("model", "gpt-4o-mini")
    temperature = float(cfg.get("temperature", 0.0))
    max_tokens = int(cfg.get("max_tokens", 200))

    provider = make_provider(ProviderConfig(provider=provider_name, model=model, temperature=temperature, max_tokens=max_tokens))

    trigger_str = "{optim_str}"
    if args.trigger:
        import json

        trig = json.loads(Path(args.trigger).read_text(encoding="utf-8"))
        trigger_str = trig.get("trigger", "") or "{optim_str}"

    run_dir = make_run_dir(prefix="eval", base_dir="results")
    logger = setup_logger(run_dir=run_dir)
    logger.info(f"Evaluating dataset={args.dataset} model={model} provider={provider_name}")

    report = eval_trigger_prompt_only(
        provider,
        args.dataset,
        trigger=trigger_str,
        limit_items=args.limit_items,
        target_contains=args.target_contains,
    )

    out_path = Path(args.out) if args.out else (run_dir / f"eval_{int(time.time())}.json")
    write_json(out_path, report)

    logger.info("Summary:")
    logger.info(str(report["summary"]))
    logger.info(f"Report saved: {out_path}")


if __name__ == "__main__":
    main()