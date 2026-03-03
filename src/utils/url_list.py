"""
utils/url_list.py

Responsibility
--------------
Load and validate curated URL lists (YAML/JSON/CSV). This replaces the original repository's
Google Custom Search + query generation pipeline. Only pre-approved URLs are processed.

Features
--------
- Validates URL scheme and hostname
- Supports per-item goals, tags, notes
- Supports injection metadata (surface + strategy) to control where {optim_str} goes in prompts
- Supports capture overrides (timeouts, wait_until, etc.)

Used by
-------
capture_dataset.py (via iter_items/load_url_list)
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Union
from urllib.parse import urlparse

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


InjectionSurface = Literal["axtree_txt", "pruned_html", "both"]
InjectionStrategy = Literal["append", "prepend", "insert_after_regex", "replace_regex"]
AnchorType = Literal["regex", "literal"]


@dataclass
class InjectionAnchor:
    type: AnchorType = "literal"
    value: str = ""


@dataclass
class InjectionSpec:
    surface: InjectionSurface = "axtree_txt"
    strategy: InjectionStrategy = "append"
    anchor: Optional[InjectionAnchor] = None


@dataclass
class CaptureOverrides:
    timeout_ms: Optional[int] = None
    wait_until: Optional[str] = None
    headless: Optional[bool] = None
    max_html_chars: Optional[int] = None


@dataclass
class UrlListItem:
    id: str
    url: str
    goal: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    injection: Optional[InjectionSpec] = None
    overrides: CaptureOverrides = field(default_factory=CaptureOverrides)


@dataclass
class UrlList:
    version: int = 1
    name: str = "url_list"
    default_goal: Optional[str] = None
    defaults: CaptureOverrides = field(default_factory=CaptureOverrides)
    items: List[UrlListItem] = field(default_factory=list)


def load_url_list(path: Union[str, Path]) -> UrlList:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"URL list not found: {path}")

    suf = path.suffix.lower()
    if suf in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("pyyaml is required for YAML url lists. pip install pyyaml")
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return _parse_structured(raw, source=str(path))

    if suf == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return _parse_structured(raw, source=str(path))

    if suf == ".csv":
        rows = _load_csv(path)
        return _parse_csv(rows, source=str(path))

    raise ValueError(f"Unsupported URL list format: {suf} (expected .yaml/.yml/.json/.csv)")


def iter_items(url_list: UrlList) -> Iterable[UrlListItem]:
    for item in url_list.items:
        if not item.goal:
            item.goal = url_list.default_goal
        item.overrides = _merge_overrides(url_list.defaults, item.overrides)
        yield item


def to_dict(item: UrlListItem) -> Dict[str, Any]:
    return asdict(item)


def _validate_url(url: str, where: str) -> None:
    if not url:
        raise ValueError(f"{where}: URL is empty")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"{where}: URL scheme must be http/https (got '{parsed.scheme}')")
    if not parsed.netloc:
        raise ValueError(f"{where}: URL missing hostname")


def _parse_structured(raw: Dict[str, Any], source: str) -> UrlList:
    if not isinstance(raw, dict):
        raise ValueError(f"{source}: expected object at top-level")

    ul = UrlList(
        version=int(raw.get("version", 1)),
        name=str(raw.get("name", "url_list")),
        default_goal=raw.get("default_goal"),
        defaults=_parse_overrides(raw.get("defaults") or {}, where=f"{source}:defaults"),
        items=[],
    )

    items_raw = raw.get("items")
    if not isinstance(items_raw, list) or not items_raw:
        raise ValueError(f"{source}: expected non-empty items list")

    seen = set()
    for i, obj in enumerate(items_raw):
        if not isinstance(obj, dict):
            raise ValueError(f"{source}:items[{i}] must be object")

        item_id = str(obj.get("id", "")).strip()
        if not item_id:
            raise ValueError(f"{source}:items[{i}] missing id")
        if item_id in seen:
            raise ValueError(f"{source}: duplicate id '{item_id}'")
        seen.add(item_id)

        url = str(obj.get("url", "")).strip()
        _validate_url(url, where=f"{source}:items[{i}].url")

        tags = obj.get("tags") or []
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split("|") if t.strip()]
        if not isinstance(tags, list):
            raise ValueError(f"{source}:items[{i}].tags must be list or string")

        injection = None
        if obj.get("injection") is not None:
            injection = _parse_injection(obj["injection"], where=f"{source}:items[{i}].injection")

        overrides = _parse_overrides(obj.get("overrides") or {}, where=f"{source}:items[{i}].overrides")

        ul.items.append(
            UrlListItem(
                id=item_id,
                url=url,
                goal=obj.get("goal"),
                tags=[str(t).strip() for t in tags if str(t).strip()],
                notes=obj.get("notes"),
                injection=injection,
                overrides=overrides,
            )
        )

    return ul


def _load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        rows = []
        for row in r:
            rows.append({(k or "").strip(): (v or "").strip() for k, v in row.items()})  # type: ignore
        return rows


def _parse_csv(rows: List[Dict[str, str]], source: str) -> UrlList:
    ul = UrlList(version=1, name=Path(source).stem)
    seen = set()

    for idx, row in enumerate(rows):
        item_id = (row.get("id") or "").strip()
        url = (row.get("url") or "").strip()

        if not item_id:
            raise ValueError(f"{source}: row {idx+2} missing id")
        if item_id in seen:
            raise ValueError(f"{source}: duplicate id '{item_id}'")
        seen.add(item_id)

        _validate_url(url, where=f"{source}: row {idx+2} url")

        tags_raw = (row.get("tags") or "").strip()
        tags = [t.strip() for t in tags_raw.split("|") if t.strip()] if tags_raw else []

        inj = _parse_injection_from_csv(row)

        ov = CaptureOverrides(
            timeout_ms=_opt_int(row.get("timeout_ms")),
            wait_until=_opt_str(row.get("wait_until")),
            headless=_opt_bool(row.get("headless")),
            max_html_chars=_opt_int(row.get("max_html_chars")),
        )

        ul.items.append(
            UrlListItem(
                id=item_id,
                url=url,
                goal=_opt_str(row.get("goal")),
                tags=tags,
                notes=_opt_str(row.get("notes")),
                injection=inj,
                overrides=ov,
            )
        )

    return ul


def _parse_injection(obj: Any, where: str) -> InjectionSpec:
    if not isinstance(obj, dict):
        raise ValueError(f"{where}: injection must be object")

    surface = obj.get("surface", "axtree_txt")
    strategy = obj.get("strategy", "append")
    if surface not in {"axtree_txt", "pruned_html", "both"}:
        raise ValueError(f"{where}: invalid surface '{surface}'")
    if strategy not in {"append", "prepend", "insert_after_regex", "replace_regex"}:
        raise ValueError(f"{where}: invalid strategy '{strategy}'")

    anchor = None
    if obj.get("anchor") is not None:
        a = obj["anchor"]
        if not isinstance(a, dict):
            raise ValueError(f"{where}: anchor must be object")
        atype = a.get("type", "literal")
        aval = str(a.get("value", "")).strip()
        if atype not in {"regex", "literal"}:
            raise ValueError(f"{where}: invalid anchor type '{atype}'")
        if not aval:
            raise ValueError(f"{where}: anchor value required")
        if atype == "regex":
            re.compile(aval)
        anchor = InjectionAnchor(type=atype, value=aval)

    return InjectionSpec(surface=surface, strategy=strategy, anchor=anchor)


def _parse_injection_from_csv(row: Dict[str, str]) -> Optional[InjectionSpec]:
    surface = (row.get("injection_surface") or "").strip()
    strategy = (row.get("injection_strategy") or "").strip()
    if not surface and not strategy:
        return None
    if not surface:
        surface = "axtree_txt"
    if not strategy:
        strategy = "append"

    if surface not in {"axtree_txt", "pruned_html", "both"}:
        raise ValueError(f"CSV: invalid injection_surface '{surface}'")
    if strategy not in {"append", "prepend", "insert_after_regex", "replace_regex"}:
        raise ValueError(f"CSV: invalid injection_strategy '{strategy}'")

    atype = _opt_str(row.get("injection_anchor_type")) or None
    aval = _opt_str(row.get("injection_anchor_value")) or None
    anchor = None
    if atype or aval:
        atype = atype or "literal"
        aval = aval or ""
        if atype not in {"regex", "literal"}:
            raise ValueError(f"CSV: invalid injection_anchor_type '{atype}'")
        if not aval:
            raise ValueError("CSV: injection_anchor_value required if anchor type set")
        if atype == "regex":
            re.compile(aval)
        anchor = InjectionAnchor(type=atype, value=aval)

    return InjectionSpec(surface=surface, strategy=strategy, anchor=anchor)


def _parse_overrides(obj: Any, where: str) -> CaptureOverrides:
    if not isinstance(obj, dict):
        raise ValueError(f"{where}: overrides must be object")
    return CaptureOverrides(
        timeout_ms=_opt_int(obj.get("timeout_ms")),
        wait_until=_opt_str(obj.get("wait_until")),
        headless=_opt_bool(obj.get("headless")),
        max_html_chars=_opt_int(obj.get("max_html_chars")),
    )


def _merge_overrides(base: CaptureOverrides, item: CaptureOverrides) -> CaptureOverrides:
    return CaptureOverrides(
        timeout_ms=item.timeout_ms if item.timeout_ms is not None else base.timeout_ms,
        wait_until=item.wait_until if item.wait_until is not None else base.wait_until,
        headless=item.headless if item.headless is not None else base.headless,
        max_html_chars=item.max_html_chars if item.max_html_chars is not None else base.max_html_chars,
    )


def _opt_str(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s or None


def _opt_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip()
    return int(s) if s else None


def _opt_bool(x: Any) -> Optional[bool]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if not s:
        return None
    if s in {"1", "true", "yes", "y"}:
        return True
    if s in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean '{x}'")