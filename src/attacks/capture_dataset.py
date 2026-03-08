#!/usr/bin/env python3
"""
capture_dataset.py (entrypoint)

Purpose
-------
Build a dataset from a curated, pre-approved list of URLs (no web search, no scraping discovery).
For each URL, this script loads the page in a Playwright Chromium browser and captures:
  - page title
  - HTML content (truncated to a configurable maximum)
  - accessibility tree snapshot (flattened to text)

Output
------
Writes a dataset folder containing:
  - src/data/datasets/<dataset_name>/items/<item_id>.json  (one JSON per captured URL)
  - src/data/datasets/<dataset_name>/meta.json            (capture metadata + list of items)

Configuration
-------------
By default uses config/dataset_capture.yaml. You can override via:
  --config <path>

Requirements
------------
- playwright (and `playwright install chromium`)
- pyyaml (if using YAML configs / URL lists)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]   # project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config  # noqa: E402
from src.utils.logging_utils import make_run_dir, save_run_metadata, setup_logger, write_json  # noqa: E402
from src.utils.url_list import load_url_list, iter_items  # noqa: E402
from src.utils.dataset_capture import capture_page, make_dataset_record, write_item_json  # noqa: E402


DEFAULT_CONFIG = "config/dataset_capture.yaml"


def main() -> None:
    ap = argparse.ArgumentParser(description="Capture dataset from curated URL list.")
    ap.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Config path (.yaml/.json). Default: {DEFAULT_CONFIG}",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)

    url_list_path = cfg.get("url_list_path", "src/data/url_lists/approved_urls.yaml")
    dataset_name = cfg.get("dataset_name", "demo_dataset")
    output_dir = Path(cfg.get("output_dir", f"src/data/datasets/{dataset_name}"))

    timeout_ms = int(cfg.get("timeout_ms", 45000))
    wait_until = str(cfg.get("wait_until", "domcontentloaded"))
    headless = bool(cfg.get("headless", True))
    max_html_chars = int(cfg.get("max_html_chars", 200000))

    run_dir = make_run_dir(prefix="capture", base_dir="results")
    logger = setup_logger(run_dir=run_dir)

    logger.info(f"Config: {args.config}")
    logger.info(f"URL list: {url_list_path}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Output: {output_dir}")

    url_list = load_url_list(url_list_path)

    meta: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "created_at_unix": int(time.time()),
        "url_list_path": str(url_list_path),
        "capture_defaults": {
            "timeout_ms": timeout_ms,
            "wait_until": wait_until,
            "headless": headless,
            "max_html_chars": max_html_chars,
        },
        "items": [],
    }

    import asyncio

    for item in iter_items(url_list):
        item_timeout = item.overrides.timeout_ms or timeout_ms
        item_wait = item.overrides.wait_until or wait_until
        item_headless = item.overrides.headless if item.overrides.headless is not None else headless
        item_max_html = item.overrides.max_html_chars or max_html_chars

        logger.info(f"Capturing {item.id}: {item.url}")
        try:
            cap = asyncio.run(
                capture_page(
                    item.url,
                    timeout_ms=item_timeout,
                    wait_until=item_wait,
                    headless=item_headless,
                    max_html_chars=item_max_html,
                )
            )
            record = make_dataset_record(item, cap)
            item_path = write_item_json(output_dir, item.id, record)
            meta["items"].append({"id": item.id, "path": str(item_path)})

        except Exception as e:
            logger.exception(f"Capture failed for {item.id}: {e}")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "meta.json", meta)
    save_run_metadata(run_dir, {"config": args.config, "output_dir": str(output_dir)})

    logger.info(f"Done. Dataset written to: {output_dir}")


if __name__ == "__main__":
    main()