"""
utils/agent_wrapper.py

Responsibility
--------------
A minimal "agent loop" wrapper:
- Builds a prompt from a captured dataset item
- Calls a Provider (OpenAI/HF) for a response
- Parses the response into an action

Also supports an optional restricted browser executor:
- Only WAIT and same-host GOTO are allowed (safety guard)

Used by
-------
run_demo.py
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from utils.action_parser import parse_action
from utils.prompt_builder import build_prompt
from utils.providers import BaseProvider


def _domain(url: str) -> str:
    return urlparse(url).netloc.lower()


@dataclass
class AgentConfig:
    mode: str = "prompt_only"  # "prompt_only" | "browser"
    headless: bool = True


class WebAgent:
    def __init__(self, provider: BaseProvider, agent_cfg: AgentConfig):
        self.provider = provider
        self.cfg = agent_cfg

    def propose_action(self, item: Dict[str, Any], trigger: str = "{optim_str}") -> Dict[str, str]:
        prompt = build_prompt(item, trigger=trigger)
        out = self.provider.complete(prompt)
        return parse_action(out)

    async def run_one_step_browser(self, start_url: str, action: Dict[str, str]) -> None:
        """
        Restrictive execution: only WAIT and GOTO to same host.
        You can expand later with careful safety gating.
        """
        from playwright.async_api import async_playwright  # local import

        allow_host = _domain(start_url)

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.cfg.headless)
            page = await browser.new_page()
            try:
                await page.goto(start_url, wait_until="domcontentloaded", timeout=45000)

                atype = action.get("type", "NONE")
                args = action.get("args", "")

                if atype == "WAIT":
                    ms = int(args) if args.isdigit() else 1000
                    await page.wait_for_timeout(ms)
                    return

                if atype == "GOTO":
                    target = args.strip()
                    if not target.startswith("http"):
                        raise ValueError("GOTO requires full URL")
                    if _domain(target) != allow_host:
                        raise ValueError(f"Blocked host {_domain(target)} (allowlisted: {allow_host})")
                    await page.goto(target, wait_until="domcontentloaded", timeout=45000)
                    return

                raise ValueError(f"Blocked action type in browser mode: {atype}")
            finally:
                await page.close()
                await browser.close()