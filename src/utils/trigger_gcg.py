"""
utils/trigger_gcg.py

Responsibility
--------------
Placeholder for a white-box (gradient-based) GCG-style trigger optimizer.
This will require a local model backend that exposes logits/gradients
(e.g., transformers HFLocalProvider plus a specialized optimizer implementation).

Current state
-------------
Stub only. It exists so:
- utils/trigger_registry.py can register it
- make_trigger.py can list it as available
- selecting it will raise a clear error

Used by
-------
src/make_trigger.py via utils/trigger_registry.py
"""

from __future__ import annotations


class GCGStub:
    name = "gcg"

    def run(self, *, cfg: dict, items: list[dict], provider) -> dict:
        raise NotImplementedError(
            "GCG optimizer not implemented yet. For now use --algo blackbox_random_search "
            "or implement the GCG optimizer in src/utils/trigger_gcg.py."
        )