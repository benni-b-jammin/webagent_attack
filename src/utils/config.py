"""
utils/config.py

Responsibility
--------------
Small helpers for configuration management:
- Load YAML/JSON files
- Read environment variables in a typed way (bool/int)
- Merge dictionaries (shallow)

Used by
-------
All entrypoints under src/ (capture_dataset, run_demo, eval_trigger, make_trigger)
"""
from __future__ import annotations

import sys
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

ROOT = Path(__file__).resolve().parents[2]   # project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_config(path: str, *, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load YAML or JSON config from disk.
    Returns {} if file is missing and a default is provided; otherwise raises.
    """
    p = Path(path)
    if not p.exists():
        if default is not None:
            return dict(default)
        raise FileNotFoundError(f"Config not found: {p}")

    suffix = p.suffix.lower()
    txt = p.read_text(encoding="utf-8")

    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("pyyaml not installed. Install with: pip install pyyaml")
        return yaml.safe_load(txt) or {}

    if suffix == ".json":
        return json.loads(txt)

    raise ValueError(f"Unsupported config format: {suffix} (expected .yaml/.yml/.json)")


def env(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Convenience wrapper for environment variables.
    """
    return os.environ.get(key, default)


def env_bool(key: str, default: bool = False) -> bool:
    v = os.environ.get(key)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(key: str, default: int) -> int:
    v = os.environ.get(key)
    if v is None:
        return default
    return int(v.strip())


def merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Shallow merge dicts: keys in b override keys in a.
    """
    out = dict(a)
    out.update(b)
    return out