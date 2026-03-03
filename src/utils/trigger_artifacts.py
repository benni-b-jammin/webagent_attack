"""
utils/trigger_artifacts.py

Responsibility
--------------
Read/write trigger artifacts to disk in a stable JSON format so they can be used across:
- make_trigger.py (create/save)
- run_demo.py (load)
- eval_trigger.py (load)

Artifact format (JSON)
----------------------
{
  "created_at_unix": 1234567890,
  "algo": "blackbox_random_search",
  "provider": {"provider": "...", "model": "...", ...},
  "dataset": "data/datasets/...",
  "trigger": "....",
  "trace": {...}   # optional
}
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class TriggerArtifact:
    created_at_unix: int
    algo: str
    trigger: str
    provider: Dict[str, Any]
    dataset: Optional[str] = None
    trace: Optional[Dict[str, Any]] = None
    config_path: Optional[str] = None


def save_trigger(path: str | Path, artifact: TriggerArtifact) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(artifact), indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def load_trigger(path: str | Path) -> TriggerArtifact:
    p = Path(path)
    obj = json.loads(p.read_text(encoding="utf-8"))
    return TriggerArtifact(
        created_at_unix=int(obj.get("created_at_unix", int(time.time()))),
        algo=str(obj.get("algo", "")),
        trigger=str(obj.get("trigger", "")),
        provider=dict(obj.get("provider", {})),
        dataset=obj.get("dataset"),
        trace=obj.get("trace"),
        config_path=obj.get("config_path"),
    )