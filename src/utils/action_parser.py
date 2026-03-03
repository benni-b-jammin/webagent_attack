"""
utils/action_parser.py

Responsibility
--------------
Parse model output into a structured action dictionary.

Expected output format
----------------------
ACTION: <GOTO|CLICK|TYPE|WAIT> <args>

If no ACTION line is found, returns type=NONE with raw text in args.

Used by
-------
agent_wrapper.py, eval_runner.py (metrics)
"""
from __future__ import annotations

import re
from typing import Dict

_ACTION_RE = re.compile(r"^\s*ACTION:\s*(GOTO|CLICK|TYPE|WAIT)\s*(.*)\s*$", re.IGNORECASE)


def parse_action(text: str) -> Dict[str, str]:
    """
    Parse:
      ACTION: CLICK text="Sign in"
    into:
      {"type": "CLICK", "args": "text=\"Sign in\""}
    """
    m = _ACTION_RE.search(text or "")
    if not m:
        return {"type": "NONE", "args": (text or "").strip()}
    return {"type": m.group(1).upper(), "args": (m.group(2) or "").strip()}


def is_action(text: str) -> bool:
    return _ACTION_RE.search(text or "") is not None