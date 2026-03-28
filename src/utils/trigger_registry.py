"""
utils/trigger_registry.py

Responsibility
--------------
Central registry mapping algorithm names to implementation callables/classes.

Used by
-------
src/make_trigger.py:
- parse --algo
- look up the optimizer in this registry
- run it to get (trigger, trace)
"""

from __future__ import annotations

from typing import Callable, Dict, Protocol, Any, List


class TriggerAlgo(Protocol):
    """
    Standard interface each trigger algorithm should implement.
    """
    name: str

    def run(self, *, cfg: dict, items: List[dict], provider) -> dict:
        """
        Returns:
          {"trigger": <str>, "trace": <dict>}
        """
        ...


_REGISTRY: Dict[str, TriggerAlgo] = {}


def register(algo: TriggerAlgo) -> None:
    _REGISTRY[algo.name] = algo


def get(algo_name: str) -> TriggerAlgo:
    if algo_name not in _REGISTRY:
        raise KeyError(f"Unknown trigger algo '{algo_name}'. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[algo_name]


def available() -> list[str]:
    return sorted(_REGISTRY.keys())


# Import side effects register algorithms here
def init_registry() -> None:
    """
    Call this once from make_trigger.py before using get().
    Keeps imports explicit and avoids __init__.py patterns.
    """
    # local imports to avoid heavy deps unless needed
    from src.utils.trigger_blackbox import BlackboxRandomSearch
    from src.utils.trigger_gcg import GCGTrigger

    register(BlackboxRandomSearch())
    register(GCGTrigger())