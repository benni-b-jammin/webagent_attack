"""
utils/dataset_capture.py

Responsibility
--------------
Capture webpage representations for a dataset using Playwright:
- page title
- HTML content (truncated)
- accessibility tree snapshot (flattened to text)

This is the core of the new dataset pipeline (curated URL ingestion).

Used by
-------
capture_dataset.py
"""
from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.url_list import UrlListItem


def flatten_a11y_tree(node: Any, depth: int = 0, lines: Optional[list[str]] = None) -> str:
    if lines is None:
        lines = []
    if not isinstance(node, dict):
        return ""
    role = node.get("role", "")
    name = node.get("name", "")
    value = node.get("value", "")
    desc = " ".join(str(x) for x in [role, name, value] if x).strip()
    if desc:
        lines.append(("  " * depth) + desc)
    for child in node.get("children", []) or []:
        flatten_a11y_tree(child, depth + 1, lines)
    return "\n".join(lines)


async def capture_page(
    url: str,
    *,
    timeout_ms: int = 45000,
    wait_until: str = "domcontentloaded",
    headless: bool = True,
    max_html_chars: int = 200000,
) -> Dict[str, Any]:
    from playwright.async_api import async_playwright  # local import

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        page = await browser.new_page()
        try:
            await page.goto(url, wait_until=wait_until, timeout=timeout_ms)
            await page.wait_for_timeout(500)

            title = await page.title()
            html = await page.content()
            if len(html) > max_html_chars:
                html = html[:max_html_chars] + "\n<!-- TRUNCATED -->"

            a11y = await page.accessibility.snapshot()
            axtree_txt = flatten_a11y_tree(a11y) if a11y else ""

            return {
                "url": url,
                "title": title,
                "pruned_html": html,
                "axtree_txt": axtree_txt,
            }
        finally:
            await page.close()
            await browser.close()


def make_dataset_record(item: UrlListItem, capture: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": item.id,
        "url": item.url,
        "goal": item.goal,
        "tags": item.tags,
        "notes": item.notes,
        "injection": asdict(item.injection) if item.injection else None,
        "captured_at_unix": int(time.time()),
        **capture,
    }


def write_item_json(out_dir: Path, record: Dict[str, Any]) -> Path:
    import json

    items_dir = out_dir / "items"
    items_dir.mkdir(parents=True, exist_ok=True)
    path = items_dir / f"{record['id']}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return path