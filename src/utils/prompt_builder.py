"""
src/utils/prompt_builder.py
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

from browsergym.core.action.highlevel import HighLevelActionSet


def apply_injection(text: str, trigger: Optional[str], strategy: str = "append") -> str:
    if not trigger:
        return text
    if strategy == "prepend":
        return f"{trigger}\n{text}"
    return f"{text}\n{trigger}"


def _normalize_observation(obs: Dict[str, Any]) -> Dict[str, Any]:
    obs = deepcopy(obs)

    obs.setdefault("goal_object", [])
    obs.setdefault("chat_messages", [])
    obs.setdefault("open_pages_urls", [])
    obs.setdefault("open_pages_titles", [])
    obs.setdefault("axtree_txt", "")
    obs.setdefault("pruned_html", "")
    obs.setdefault("last_action_error", "")
    obs.setdefault("last_action", "")
    obs.setdefault("action_history", [])
    obs.setdefault("active_page_index", 0)

    return obs


def _make_action_space_text() -> str:
    action_set = HighLevelActionSet(
        subsets=["chat", "tab", "nav", "bid", "infeas"],
        strict=False,
        multiaction=False,
        demo_mode=False,
    )

    return """# Action Space

""" + action_set.describe(with_long_description=False, with_examples=False) + """

Only a single action can be provided at once.

Valid examples:
```click("12")```
```fill("2819", "Product code: 1ss2421gg3")```
```send_msg_to_user("There is no "cart" on this page.")```
"""


def _goal_lines_from_goal_object(obs: Dict[str, Any]) -> List[str]:
    lines = []
    for item in obs.get("goal_object", []):
        if isinstance(item, str):
            lines.append(item)
        elif isinstance(item, dict):
            if item.get("type") == "text":
                lines.append(str(item.get("text", "")))
            else:
                lines.append(str(item))
        else:
            lines.append(str(item))
    return lines


def _goal_lines_from_chat_messages(obs: Dict[str, Any]) -> List[str]:
    lines = []
    for msg in obs.get("chat_messages", []):
        role = msg.get("role")
        if role == "user":
            lines.append(str(msg.get("message", "")))
        elif role == "user_image":
            lines.append("[user provided an image]")
    return lines


def _build_messages_common(
    obs: Dict[str, Any],
    *,
    goal_lines: List[str],
    include_html: bool,
    include_axtree: bool,
) -> List[Dict[str, str]]:
    system_msgs = [
        {
            "type": "text",
            "text": """# Instructions

Review the current state of the page and all other information to find the best
possible next action to accomplish your goal. Your answer will be interpreted
and executed by a program, make sure to follow the formatting instructions.
""",
        }
    ]

    user_msgs = [{"type": "text", "text": "# Goal\n"}]

    for line in goal_lines:
        user_msgs.append({"type": "text", "text": str(line)})

    user_msgs.append({"type": "text", "text": "# Currently open tabs\n"})

    active_page_index = obs.get("active_page_index", 0)
    for page_index, (page_url, page_title) in enumerate(
        zip(obs.get("open_pages_urls", []), obs.get("open_pages_titles", []))
    ):
        active = " (active tab)" if page_index == active_page_index else ""
        user_msgs.append(
            {
                "type": "text",
                "text": f"""Tab {page_index}{active}
Title: {page_title}
URL: {page_url}
""",
            }
        )

    if include_axtree:
        user_msgs.append(
            {
                "type": "text",
                "text": f"""# Current page Accessibility Tree

{obs.get("axtree_txt", "")}

""",
            }
        )

    if include_html:
        user_msgs.append(
            {
                "type": "text",
                "text": f"""# Current page DOM

{obs.get("pruned_html", "")}

""",
            }
        )

    user_msgs.append({"type": "text", "text": _make_action_space_text()})

    user_msgs.append(
        {
            "type": "text",
            "text": """# Important bid rule

When selecting an element, you must use the bid shown in square brackets in the Accessibility Tree, such as [1253].
Do not use attribute values like controls='vkWLu', names, labels, placeholders, or example bids from the action documentation.
Only use bids that appear in square brackets [] in the current page Accessibility Tree.
""",
        }
    )

    if obs.get("action_history"):
        user_msgs.append({"type": "text", "text": "# History of past actions\n"})
        for action in obs["action_history"]:
            user_msgs.append({"type": "text", "text": str(action)})

    if obs.get("last_action_error"):
        user_msgs.append(
            {
                "type": "text",
                "text": f"""# Error message from last action

{obs["last_action_error"]}
""",
            }
        )

    user_msgs.append(
        {
            "type": "text",
            "text": """# Next action

You will now think step by step and produce your next best action. Reflect on
your past actions, any resulting error message, and the current state of the
page before deciding on your next action. If you believe you are done with the
task, please produce a noop.
""",
        }
    )

    sys_content = "\n".join(m["text"] for m in system_msgs)
    user_content = "\n".join(m["text"] for m in user_msgs)

    return [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": user_content},
    ]


def _build_chat_mode_messages(
    obs: Dict[str, Any],
    *,
    include_html: bool,
    include_axtree: bool,
) -> List[Dict[str, str]]:
    goal_lines = _goal_lines_from_chat_messages(obs)
    return _build_messages_common(
        obs,
        goal_lines=goal_lines,
        include_html=include_html,
        include_axtree=include_axtree,
    )


def _build_headless_messages(
    obs: Dict[str, Any],
    *,
    include_html: bool,
    include_axtree: bool,
) -> List[Dict[str, str]]:
    goal_lines = _goal_lines_from_goal_object(obs)
    return _build_messages_common(
        obs,
        goal_lines=goal_lines,
        include_html=include_html,
        include_axtree=include_axtree,
    )


def build_messages(
    observation: Dict[str, Any],
    *,
    trigger: Optional[str] = None,
    include_html: bool = False,
    include_axtree: bool = True,
    chat_mode: bool = False,
) -> List[Dict[str, str]]:
    obs = _normalize_observation(observation)

    inj = obs.get("injection") or {}
    surface = (inj.get("surface") or "axtree_txt").strip()
    strategy = (inj.get("strategy") or "append").strip()

    # Trigger injection site:
    # default to AXTree, the original repo's placeholder-replacement workflow.
    if trigger:
        if include_axtree and surface in {"axtree_txt", "both"}:
            obs["axtree_txt"] = apply_injection(obs.get("axtree_txt", ""), trigger, strategy)

        if include_html and surface in {"pruned_html", "both"}:
            obs["pruned_html"] = apply_injection(obs.get("pruned_html", ""), trigger, strategy)

    if chat_mode:
        return _build_chat_mode_messages(
            obs,
            include_html=include_html,
            include_axtree=include_axtree,
        )

    return _build_headless_messages(
        obs,
        include_html=include_html,
        include_axtree=include_axtree,
    )


def build_prompt(
    observation: Dict[str, Any],
    *,
    trigger: Optional[str] = None,
    include_html: bool = False,
    include_axtree: bool = True,
    chat_mode: bool = False,
) -> str:
    messages = build_messages(
        observation,
        trigger=trigger,
        include_html=include_html,
        include_axtree=include_axtree,
        chat_mode=chat_mode,
    )
    return "\n\n".join(f"{m['role'].upper()}:\n{m['content']}" for m in messages)