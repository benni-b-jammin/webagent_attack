"""
utils/eval_runner.py

Responsibility
--------------
Run prompt-only evaluations over a dataset:
- For each item, build prompt with injected trigger
- Query the provider
- Compute basic metrics (ACTION line present; optional substring target match)
- Return report dict with summary + per-item results

Used by
-------
eval_trigger.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.action_parser import is_action
from src.utils.prompt_builder import build_prompt
from src.utils.providers import BaseProvider


def load_dataset_items(dataset_dir: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    d = Path(dataset_dir) / "items"
    if not d.exists():
        raise FileNotFoundError(f"Dataset items dir not found: {d}")
    files = sorted(d.glob("*.json"))
    if limit is not None:
        files = files[:limit]
    return [json.loads(p.read_text(encoding="utf-8")) for p in files]


def eval_trigger_prompt_only(
    provider: BaseProvider,
    dataset_dir: str,
    *,
    trigger: str,
    limit_items: Optional[int] = None,
    target_contains: Optional[str] = None,
) -> Dict[str, Any]:
    items = load_dataset_items(dataset_dir, limit=limit_items)

    results = []
    n_action = 0
    n_target = 0
    tgt = target_contains.lower() if target_contains else None

    for it in items:
        prompt = build_prompt(it, trigger=trigger)
        out = provider.complete(prompt)

        has_action = is_action(out)
        if has_action:
            n_action += 1

        has_target = False
        if tgt:
            has_target = (tgt in (out or "").lower())
            if has_target:
                n_target += 1

        results.append(
            {
                "id": it.get("id"),
                "url": it.get("url"),
                "has_action": has_action,
                "has_target": has_target,
                "output": out,
            }
        )

    summary = {
        "dataset": dataset_dir,
        "items": len(items),
        "action_rate": n_action / max(1, len(items)),
        "target_contains": target_contains,
        "target_rate": (n_target / max(1, len(items))) if tgt else None,
    }

    return {"summary": summary, "results": results}