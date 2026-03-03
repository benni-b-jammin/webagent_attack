"""
utils/trigger_blackbox.py

Responsibility
--------------
Black-box trigger search that works with API models (OpenAI, HF hosted, etc.).
No gradients required.

Algorithm (simple baseline)
---------------------------
Random search over candidate strings:
- sample candidate trigger strings
- for a small subset of dataset items:
    prompt = prompt_builder.build_prompt(item, trigger=candidate)
    output = provider.complete(prompt)
    score += 1 if target substring appears in output else 0
- keep best candidate by average score

Used by
-------
src/make_trigger.py via utils/trigger_registry.py
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

from utils.prompt_builder import build_prompt


class BlackboxRandomSearch:
    name = "blackbox_random_search"

    def run(self, *, cfg: dict, items: List[dict], provider) -> dict:
        alphabet = cfg.get("alphabet", "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ")
        trigger_len = int(cfg.get("trigger_len", 12))
        iters = int(cfg.get("iters", 50))
        score_items = int(cfg.get("score_items", 3))
        target_contains = str(cfg.get("target_contains", "ACTION:"))

        subset = items[: min(len(items), score_items)]
        if not subset:
            return {"trigger": "", "trace": {"error": "no_items"}}

        best_trigger = ""
        best_score = -1.0
        history = []

        for i in range(iters):
            cand = "".join(random.choice(alphabet) for _ in range(trigger_len))
            scores = []

            for it in subset:
                prompt = build_prompt(it, trigger=cand)
                out = provider.complete(prompt)
                scores.append(1.0 if target_contains.lower() in (out or "").lower() else 0.0)

            avg = sum(scores) / max(1, len(scores))
            history.append({"iter": i, "cand": cand, "avg_score": avg})

            if avg > best_score:
                best_score = avg
                best_trigger = cand

        return {
            "trigger": best_trigger,
            "trace": {"algo": self.name, "best_score": best_score, "history": history},
        }