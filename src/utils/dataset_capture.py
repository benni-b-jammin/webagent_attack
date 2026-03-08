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


def _apply_placeholder(text: str, placeholder: str, strategy: str) -> str:
    """
    Insert placeholder into text according to strategy.
    Supported: prepend, append
    """
    if not placeholder:
        return text
    if strategy == "prepend":
        return f"{placeholder} {text}" if text else placeholder
    # default append
    return f"{text} {placeholder}" if text else placeholder


def make_dataset_record(
    item: UrlListItem,
    capture: Dict[str, Any],
    *,
    placeholder: str = "{optim_str}",
    force_placeholder_in_axtree: bool = True,
) -> Dict[str, Any]:
    """
    Produce an observation dict compatible with the *original repo* style:
      - goal_object: list[str]
      - open_pages_urls: list[str]
      - open_pages_titles: list[str]
      - axtree_txt: str
      - pruned_html: str

    Placeholder insertion
    ---------------------
    - If item.injection.surface includes axtree_txt, insert placeholder into axtree_txt
    - If item.injection.surface includes pruned_html, insert placeholder into pruned_html
    - If force_placeholder_in_axtree=True, also insert into axtree_txt even if injection.surface isn't axtree surface.
      (This matches your desired example output.)
    """
    url = capture.get("url", item.url)
    title = capture.get("title", "")
    axtree_txt = capture.get("axtree_txt", "") or ""
    pruned_html = capture.get("pruned_html", "") or ""

    strategy = "prepend"
    surface = "axtree_txt"
    if item.injection:
        strategy = item.injection.strategy or "append"
        surface = item.injection.surface or "axtree_txt"

    # Apply placeholder based on injection metadata
    if item.injection:
        if surface in ("axtree_txt", "both"):
            axtree_txt = _apply_placeholder(axtree_txt, placeholder, strategy)
        if surface in ("pruned_html", "both"):
            pruned_html = _apply_placeholder(pruned_html, placeholder, strategy)

    # Optional: ensure placeholder is present in axtree regardless (recommended if your attack surface is a11y tree)
    if force_placeholder_in_axtree and placeholder not in axtree_txt:
        axtree_txt = _apply_placeholder(axtree_txt, placeholder, "prepend")

    return {
        "goal_object": [item.goal] if item.goal else [],
        "open_pages_urls": [url],
        "open_pages_titles": [title],
        "axtree_txt": axtree_txt,
        "pruned_html": pruned_html,
    }


def write_item_json(out_dir: Path, item_id: str, record: Dict[str, Any]) -> Path:
    """
    Write a single captured observation JSON in the new (original-repo-style) schema.

    Since the schema no longer includes an 'id' field, the caller must provide item_id
    (typically the UrlListItem.id) for naming the file deterministically.
    """
    import json

    items_dir = out_dir / "items"
    items_dir.mkdir(parents=True, exist_ok=True)
    path = items_dir / f"{item_id}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return path