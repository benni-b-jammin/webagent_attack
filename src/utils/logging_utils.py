"""
utils/logging_utils.py

Responsibility
--------------
Centralized logging and run-folder utilities:
- Create timestamped run directories under results/
- Set up a console + file logger
- Helpers to write JSON/text files
- Save run metadata for reproducibility

Used by
-------
capture_dataset.py, eval_trigger.py, make_trigger.py
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional


def make_run_dir(prefix: str = "run", base_dir: str = "results") -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"{prefix}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_logger(name: str = "webagent", run_dir: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if run_dir is not None:
        fh = logging.FileHandler(run_dir / "log.txt", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_run_metadata(run_dir: Path, meta: Dict[str, Any]) -> None:
    write_json(run_dir / "meta.json", meta)