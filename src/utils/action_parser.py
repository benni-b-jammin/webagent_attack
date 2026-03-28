"""
src/utils/action_parser.py

Responsibility
--------------
Extract a BrowserGym high-level action string from model output.

Supports the original repo's action space.
"""
from __future__ import annotations

import re


_FENCED_CODE_RE = re.compile(
    r"```(?:python)?\s*(.*?)\s*```",
    re.DOTALL | re.IGNORECASE,
)

_ACTION_CALL_RE = re.compile(
    r"""
    (?P<action>
        (?:
            noop|
            send_msg_to_user|
            tab_close|
            tab_focus|
            new_tab|
            go_back|
            go_forward|
            goto|
            scroll|
            fill|
            select_option|
            click|
            dblclick|
            hover|
            press|
            focus|
            clear|
            drag_and_drop|
            upload_file|
            report_infeasible
        )
        \s*\(
            .*?
        \)
    )
    """,
    re.DOTALL | re.VERBOSE | re.IGNORECASE,
)


def extract_browsergym_action(text: str) -> str:
    """
    Extract the first BrowserGym-style action call from model output.

    Returns a raw action string such as:
        click("12")
        fill("3989", "Ich bin ein Berliner")
        noop()

    Falls back to noop() if no action is found.
    """
    if not text:
        return "noop()"

    text = text.strip()

    fenced = _FENCED_CODE_RE.findall(text)
    for block in fenced:
        match = _ACTION_CALL_RE.search(block.strip())
        if match:
            return match.group("action").strip()

    match = _ACTION_CALL_RE.search(text)
    if match:
        return match.group("action").strip()

    return "noop()"


def is_action(text: str) -> bool:
    return extract_browsergym_action(text) != "noop()"