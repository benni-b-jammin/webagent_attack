"""
src/utils/agent_wrapper.py

Responsibility
--------------
BrowserGym-compatible agent wrapper.

- preprocesses raw BrowserGym observations
- builds a BrowserGym-style prompt
- calls the LLM
- extracts a BrowserGym high-level action string
- exposes BrowserGym's action_set
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html

from src.utils.action_parser import extract_browsergym_action
from src.utils.prompt_builder import build_messages
from src.utils.providers import BaseProvider


@dataclass
class AgentConfig:
    mode: str = "browser"
    headless: bool = True
    chat_mode: bool = False
    demo_mode: str = "default"
    use_html: bool = False
    use_axtree: bool = True
    use_screenshot: bool = False


class WebAgent:
    def __init__(self, provider: BaseProvider, agent_cfg: AgentConfig):
        self.provider = provider
        self.cfg = agent_cfg
        self.stop_after_send = False

        self.action_set = HighLevelActionSet(
            subsets=["chat", "tab", "nav", "bid", "infeas"],
            strict=False,
            multiaction=False,
            demo_mode=self.cfg.demo_mode,
        )

        self.action_history = []

    def obs_preprocessor(self, obs: dict) -> dict:
        raw_goal_object = obs.get("goal_object", [])
        normalized_goal_object = []

        for item in raw_goal_object:
            if isinstance(item, str):
                normalized_goal_object.append(item)
            elif isinstance(item, dict) and "text" in item:
                normalized_goal_object.append(str(item["text"]))
            else:
                normalized_goal_object.append(str(item))

        processed_obs = {
            "chat_messages": obs.get("chat_messages", []),
            "goal_object": normalized_goal_object,
            "last_action": obs.get("last_action"),
            "last_action_error": obs.get("last_action_error", ""),
            "open_pages_urls": [str(x) for x in obs.get("open_pages_urls", [])],
            "open_pages_titles": [str(x) for x in obs.get("open_pages_titles", [])],
            "axtree_txt": flatten_axtree_to_str(obs["axtree_object"]) if "axtree_object" in obs else "",
            "pruned_html": prune_html(flatten_dom_to_str(obs["dom_object"])) if "dom_object" in obs else "",
            "active_page_index": obs.get("active_page_index", 0),
            "action_history": list(self.action_history),
        }
        print("\n=== DEBUG AXTREE PREVIEW ===")
        print(processed_obs["axtree_txt"][:2000])
        return processed_obs

    def propose_action(
            self,
            observation: Dict[str, Any],
            trigger: Optional[str] = None,
    ) -> str:
        # Build prompt with a placeholder first, mirroring the original repo style.
        messages = build_messages(
            observation,
            trigger="{optim_str}" if trigger else None,
            include_html=self.cfg.use_html,
            include_axtree=self.cfg.use_axtree,
            chat_mode=self.cfg.chat_mode,
        )

        # Replace placeholder with actual trigger right before inference.
        if trigger:
            patched_messages = []
            for m in messages:
                content = m["content"]
                if isinstance(content, str):
                    content = content.replace("{optim_str}", trigger)
                patched_messages.append(
                    {
                        "role": m["role"],
                        "content": content,
                    }
                )
            messages = patched_messages

        print("\n=== FULL PROMPT SENT TO MODEL ===")
        for m in messages:
            print(f"\n--- {m['role'].upper()} ---\n{m['content']}")

        raw_output = self.provider.complete_messages(messages)

        print("\n=== RAW MODEL OUTPUT ===")
        print(raw_output)

        action = extract_browsergym_action(raw_output)

        print("\n=== EXTRACTED ACTION ===")
        print(action)

        self.action_history.append(action)
        return action

    def act(self, observation: Dict[str, Any]) -> str:
        return self.propose_action(observation, trigger=None)

    def get_action(self, obs: Dict[str, Any]) -> tuple[Optional[str], dict]:
        if getattr(self, "stop_after_send", False):
            self.stop_after_send = False
            return None, {}

        action = self.propose_action(obs, trigger=None)

        if action is None:
            return None, {}

        action_str = str(action).strip()

        if action_str == "noop()" or "I'm done" in action_str:
            return None, {}

        if action_str.startswith("send_msg_to_user("):
            self.stop_after_send = True
            return action_str, {}

        return action_str, {}