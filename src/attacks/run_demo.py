#!/usr/bin/env python3
"""
run_demo.py (config-driven BrowserGym experiment runner)

Purpose
-------
Run a BrowserGym openended task using a config file, while preserving
the same overall control flow as the original Manipulating-Web-Agents
`run_demo.py`.

Key design
----------
- Load all settings from YAML/JSON config
- Create a BrowserGym-compatible "agent args" object
- Let BrowserGym instantiate the actual agent through `make_agent()`
- Support optional trigger JSON
- Return raw BrowserGym high-level action strings from the agent
"""

from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import dataclass
from typing import Optional

from browsergym.experiments import EnvArgs, ExpArgs, get_exp_result

from src.utils.config import load_config
from src.utils.providers import ProviderConfig, make_provider
from src.utils.agent_wrapper import AgentConfig, WebAgent


DEFAULT_CONFIG = "src/config/demo_runs/demo_default.yaml"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run BrowserGym experiment from config."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Path to config YAML/JSON (default: {DEFAULT_CONFIG})",
    )

    parser.add_argument("--provider_name", help="Override provider.name from config")
    parser.add_argument("--provider_model", help="Override provider.model from config")
    parser.add_argument("--provider_temperature", type=float, help="Override provider.temperature from config")
    parser.add_argument("--provider_max_tokens", type=int, help="Override provider.max_tokens from config")

    parser.add_argument("--headless", help="Override env.headless (true/false)")
    parser.add_argument("--trigger_path", help="Override trigger.path from config")

    parser.add_argument("--start_url", help="Override env.start_url from config")
    parser.add_argument("--goal", help="Override env.goal from config")
    parser.add_argument("--n_steps", type=int, help="Override env.n_steps from config")
    parser.add_argument("--exp_name", help="Override experiment.name from config")

    return parser.parse_args()


def as_bool(value, default=False):
    """
    Robust bool parsing for YAML/JSON values.
    Avoids issues like bool('false') == True.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"Cannot interpret as bool: {value!r}")


def load_trigger(trigger_path: Optional[str]) -> Optional[str]:
    """
    Load trigger string from a JSON file if provided.
    Expected schema:
        {"trigger": "..."}
    """
    if not trigger_path:
        return None

    path = pathlib.Path(trigger_path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    trigger = data.get("trigger")
    if trigger is None:
        raise KeyError(f"No 'trigger' field found in trigger JSON: {path}")
    return trigger

class BrowserGymWebAgent:
    """
    Thin adapter that BrowserGym can call.
    """

    def __init__(
        self,
        provider_cfg: ProviderConfig,
        agent_cfg: AgentConfig,
        trigger: Optional[str],
    ):
        self.provider = make_provider(provider_cfg)
        self.agent = WebAgent(self.provider, agent_cfg)
        self.trigger = trigger

        self.action_set = self.agent.action_set
        self.obs_preprocessor = self.agent.obs_preprocessor

    def get_action(self, obs):
        action = self.agent.propose_action(obs, trigger=self.trigger)

        if action is None:
            return None, {}

        action_str = str(action).strip()

        # Match original repo behavior exactly.
        if "I'm done" in action_str or "noop()" in action_str:
            return None, {}

        return action_str, {}

    def act(self, observation):
        action, _ = self.get_action(observation)
        return action

@dataclass
class BrowserGymWebAgentArgs:
    """
    BrowserGym-style agent args object.

    This mirrors the original repo pattern more closely:
    ExpArgs receives a lightweight config object, and BrowserGym
    instantiates the actual agent through `make_agent()`.
    """
    provider_cfg: ProviderConfig
    agent_cfg: AgentConfig
    trigger: Optional[str] = None

    def make_agent(self):
        return BrowserGymWebAgent(
            provider_cfg=self.provider_cfg,
            agent_cfg=self.agent_cfg,
            trigger=self.trigger,
        )


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Ensure sections exist
    cfg.setdefault("provider", {})
    cfg.setdefault("agent", {})
    cfg.setdefault("env", {})
    cfg.setdefault("experiment", {})
    cfg.setdefault("trigger", {})

    # -------------------------
    # CLI overrides
    # -------------------------
    if args.provider_name is not None:
        cfg["provider"]["name"] = args.provider_name

    if args.provider_model is not None:
        cfg["provider"]["model"] = args.provider_model

    if args.provider_temperature is not None:
        cfg["provider"]["temperature"] = args.provider_temperature

    if args.provider_max_tokens is not None:
        cfg["provider"]["max_tokens"] = args.provider_max_tokens

    if args.headless is not None:
        cfg["env"]["headless"] = as_bool(args.headless)

    if args.trigger_path is not None:
        cfg["trigger"]["path"] = args.trigger_path

    if args.start_url is not None:
        cfg["env"]["start_url"] = args.start_url

    if args.goal is not None:
        cfg["env"]["goal"] = args.goal

    if args.n_steps is not None:
        cfg["env"]["n_steps"] = args.n_steps

    if args.exp_name is not None:
        cfg["experiment"]["name"] = args.exp_name

    # -------------------------
    # Config sections
    # -------------------------
    provider_section = cfg.get("provider", {})
    agent_section = cfg.get("agent", {})
    env_section = cfg.get("env", {})
    exp_section = cfg.get("experiment", {})
    trigger_section = cfg.get("trigger", {})

    # -------------------------
    # Trigger
    # -------------------------
    trigger = load_trigger(trigger_section.get("path"))

    # -------------------------
    # Provider config
    # -------------------------
    provider_cfg = ProviderConfig(
        provider=provider_section.get("name", "openai"),
        model=provider_section.get("model", "gpt-4o-mini"),
        temperature=float(provider_section.get("temperature", 0.0)),
        max_tokens=int(provider_section.get("max_tokens", 200)),
    )

    # -------------------------
    # Shared headless setting
    # -------------------------
    headless = as_bool(
        env_section.get("headless", agent_section.get("headless", True)),
        default=True,
    )

    # -------------------------
    # Agent config
    # -------------------------
    # Match the original repo's defaults more closely.
    chat_mode = not headless

    agent_cfg = AgentConfig(
        mode="browser",
        headless=headless,
        chat_mode=chat_mode,
        demo_mode=agent_section.get("demo_mode", "default"),
        use_html=as_bool(agent_section.get("use_html", False), default=False),
        use_axtree=as_bool(agent_section.get("use_axtree", True), default=True),
        use_screenshot=as_bool(agent_section.get("use_screenshot", False), default=False),
    )

    agent_args = BrowserGymWebAgentArgs(
        provider_cfg=provider_cfg,
        agent_cfg=agent_cfg,
        trigger=trigger,
    )

    # -------------------------
    # Environment config
    # -------------------------
    env_args = EnvArgs(
        task_name="openended",
        task_seed=env_section.get("task_seed"),
        max_steps=int(env_section.get("n_steps", 1)),
        record_video=as_bool(env_section.get("record_video", True), default=True),
        headless=headless,
    )

    start_url = env_section.get("start_url")
    goal = env_section.get("goal")

    if not start_url:
        raise ValueError("env.start_url must be provided in the config.")

    if headless and not goal:
        raise ValueError("env.goal must be provided when running headless.")

    # Match original behavior:
    # - non-headless: Browser waits for user message, no goal in task_kwargs
    # - headless: include goal directly
    if not headless:
        env_args.wait_for_user_message = True
        env_args.task_kwargs = {
            "start_url": start_url,
        }
    else:
        env_args.task_kwargs = {
            "start_url": start_url,
            "goal": goal,
        }

    # -------------------------
    # Experiment config
    # -------------------------
    config_stem = pathlib.Path(args.config).stem

    exp_args = ExpArgs(
        env_args=env_args,
        agent_args=agent_args,
        exp_name=exp_section.get("name", config_stem),
    )

    results_dir = exp_section.get("results_dir", "./results")

    # -------------------------
    # Run experiment
    # -------------------------
    exp_args.prepare(results_dir)
    exp_args.run()

    # -------------------------
    # Load and print results
    # -------------------------
    exp_result = get_exp_result(exp_args.exp_dir)
    exp_record = exp_result.get_exp_record()

    print("\n=== EXPERIMENT RESULT ===")
    for key, val in exp_record.items():
        print(f"{key}: {val}")


if __name__ == "__main__":
    main()