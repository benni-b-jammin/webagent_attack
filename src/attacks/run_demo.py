#!/usr/bin/env python3
"""
run_demo.py (entrypoint)

Purpose
-------
Run a single-step "demo" of the agent loop:
  dataset item (page snapshot) -> prompt -> LLM -> parse action.

Default mode is "prompt_only": the script DOES NOT execute actions in a real browser.
Optional mode "browser" is supported, but is intentionally restrictive and only allows:
  - WAIT
  - GOTO to the same hostname as the starting URL

Inputs
------
- A dataset item JSON produced by capture_dataset.py
- Optional trigger artifact JSON (ignored if absent)

Output
------
Prints model output and the parsed action. In browser mode, also performs the restricted action.

Configuration
-------------
Defaults to configs/demo_run.yaml. Override with:
  --config <path>

Requirements
------------
- openai + OPENAI_API_KEY in environment (for provider=openai)
- playwright (only if mode=browser)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.config import load_config  # noqa: E402
from utils.providers import ProviderConfig, make_provider  # noqa: E402
from utils.agent_wrapper import AgentConfig, WebAgent  # noqa: E402


DEFAULT_CONFIG = "configs/demo_run.yaml"


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one demo step (prompt-only by default).")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help=f"Config path. Default: {DEFAULT_CONFIG}")
    ap.add_argument(
        "--dataset_item",
        default="data/datasets/demo_dataset/items/wiki_llm.json",
        help="Path to a captured dataset item JSON.",
    )
    ap.add_argument("--trigger", default=None, help="Optional trigger JSON artifact containing `trigger`.")
    ap.add_argument("--mode", choices=["prompt_only", "browser"], default="prompt_only", help="Execution mode.")
    args = ap.parse_args()

    cfg = load_config(args.config)

    provider_name = cfg.get("provider", "openai")
    model = cfg.get("model", "gpt-4o-mini")
    temperature = float(cfg.get("temperature", 0.0))
    max_tokens = int(cfg.get("max_tokens", 200))
    headless = bool(cfg.get("headless", True))

    provider = make_provider(ProviderConfig(provider=provider_name, model=model, temperature=temperature, max_tokens=max_tokens))
    agent = WebAgent(provider, AgentConfig(mode=args.mode, headless=headless))

    item = json.loads(Path(args.dataset_item).read_text(encoding="utf-8"))

    trigger_str = "{optim_str}"
    if args.trigger:
        trig = json.loads(Path(args.trigger).read_text(encoding="utf-8"))
        trigger_str = trig.get("trigger", "") or "{optim_str}"

    action = agent.propose_action(item, trigger=trigger_str)

    print("\n=== PARSED ACTION ===")
    print(action)

    if args.mode == "browser":
        import asyncio

        asyncio.run(agent.run_one_step_browser(item.get("url", ""), action))


if __name__ == "__main__":
    main()