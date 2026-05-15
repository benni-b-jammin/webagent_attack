import json
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import playwright.sync_api
from serpapi import GoogleSearch

from browsergym.core import _get_global_playwright
from browsergym.core.env import BrowserEnv, Chat
from browsergym.core.task import OpenEndedTask
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html

from src.utils.assets import init_script
from src.utils.data_processing import generate_search_queries


# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_URLS_PER_QUERY = 10
DEFAULT_TARGET_WEBSITES = 50
DEFAULT_MAX_CAPTURE_ATTEMPTS = 150
DEFAULT_START_URL = "https://www.google.com"


# ============================================================
# Helpers for stable website naming
# ============================================================

KNOWN_SITE_TAGS = [
    ("translate.google.com", "/?sl=", "google_translate"),
    ("www.google.com", "", "google_search"),
    ("google.com", "", "google_search"),
    ("www.ecosia.org", "", "ecosia"),
    ("ecosia.org", "", "ecosia"),
    ("www.linkedin.com", "/login", "linkedin_login"),
    ("linkedin.com", "/login", "linkedin_login"),
    ("whentowork.com", "/logins.htm", "w2w_login"),
]


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def derive_site_tag(url: str, title: str) -> str:
    """
    Produces a stable root name like 'linkedin_login' for use across the pipeline.
    """
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    path = parsed.path.lower()
    query = parsed.query.lower()

    for known_domain, known_path_fragment, tag in KNOWN_SITE_TAGS:
        if known_domain in domain:
            if known_path_fragment:
                if known_path_fragment in path or known_path_fragment in ("?" + query):
                    return tag
            else:
                return tag

    # Fallback: domain label + short title hint
    domain_parts = domain.split(".")
    root_domain = domain_parts[-2] if len(domain_parts) >= 2 else domain_parts[0]

    title_slug = slugify(title)
    if title_slug:
        title_words = title_slug.split("_")[:2]
        title_hint = "_".join(title_words)
    else:
        title_hint = "page"

    combined = f"{root_domain}_{title_hint}"
    return slugify(combined)


def make_capture_id(out_dir: Path, site_tag: str) -> str:
    """
    Creates a unique filename stem while preserving a stable root site_tag.
    """
    candidate = site_tag
    i = 2
    while (out_dir / f"{candidate}.json").exists():
        candidate = f"{site_tag}__{i}"
        i += 1
    return candidate


# ============================================================
# BrowserGym downloader
# ============================================================

class DownloaderEnv(BrowserEnv):
    """
    Minimal BrowserGym-based environment for downloading website observation JSONs.
    """
    metadata = {"render_modes": None}

    def __init__(self, start_url: str) -> None:
        self.start_url = start_url
        super().__init__(OpenEndedTask, {"start_url": start_url})

        pw: playwright.sync_api.Playwright = _get_global_playwright()
        pw.selectors.set_test_id_attribute("bid")

        self.browser = pw.chromium.launch(
            headless=self.headless,
            **self.pw_chromium_kwargs,
        )
        self.context = self.browser.new_context(
            **self.pw_context_kwargs,
        )
        self.context.expose_binding(
            "browsergym_page_activated",
            lambda source: self._activate_page_from_js(source["page"]),
        )
        self.context.add_init_script(init_script)

        self.chat = Chat(
            headless=self.headless,
            chat_size=(500, 500, 800),
        )

        self.chat.add_message(
            role="assistant",
            msg="Hi! I am your UI assistant, I can perform web tasks for you. What can I help you with?",
        )
        self.goal_object = []
        self.last_action = ""
        self.last_action_error = ""

    def obs_from_url(self, url: str) -> dict:
        self.start_time = time.time()
        self.start_url = url
        self.page = self.context.new_page()
        self.page.goto(self.start_url, timeout=15000)
        self._wait_dom_loaded()
        self._active_page_check()
        time.sleep(5)
        obs = self._get_obs()
        obs = self.obs_preprocessor(obs)
        self.page.close()
        return obs

    def obs_preprocessor(self, obs: dict) -> dict:
        return {
            "chat_messages": obs["chat_messages"],
            "goal_object": obs["goal_object"],
            "last_action": obs["last_action"],
            "last_action_error": obs["last_action_error"],
            "open_pages_urls": obs["open_pages_urls"],
            "open_pages_titles": obs["open_pages_titles"],
            "axtree_txt": flatten_axtree_to_str(obs["axtree_object"]),
            "pruned_html": prune_html(flatten_dom_to_str(obs["dom_object"])),
        }


# ============================================================
# Search
# ============================================================

def fetch_sites_from_serpapi(query: str, num_results: int = DEFAULT_URLS_PER_QUERY) -> List[str]:
    """
    Fetches a list of URLs from SerpApi Google search results.
    """
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        raise EnvironmentError("SERPAPI_KEY is not set.")

    urls: List[str] = []
    start = 0

    while len(urls) < num_results:
        params = {
            "engine": "google",
            "q": query,
            "num": min(10, num_results - len(urls)),
            "start": start,
            "api_key": api_key,
        }

        results = GoogleSearch(params).get_dict()
        organic = results.get("organic_results", [])

        if not organic:
            break

        for item in organic:
            link = item.get("link")
            if link and link not in urls:
                urls.append(link)
                if len(urls) >= num_results:
                    break

        start += 10

    return urls


def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


# ============================================================
# Main capture routine
# ============================================================

def save_observation(obs: Dict, source_query: str, out_dir: Path) -> str:
    """
    Saves a captured observation JSON and returns the saved filename stem.
    """
    url = obs["open_pages_urls"][0]
    title = obs["open_pages_titles"][0] if obs["open_pages_titles"] else "untitled"

    site_tag = derive_site_tag(url, title)
    capture_id = make_capture_id(out_dir, site_tag)

    if not obs.get("goal_object"):
        obs["goal_object"] = ["Browse the page and summarize the main purpose in one sentence."]

    obs["site_tag"] = site_tag
    obs["capture_id"] = capture_id
    obs["source_query"] = source_query

    with open(out_dir / f"{capture_id}.json", "w", encoding="utf-8") as f:
        json.dump(obs, f, indent=4)

    return capture_id


def get_website_data(
    query_types_file: str,
    n_websites: int = DEFAULT_TARGET_WEBSITES,
    n_search_queries: int = 12,
    urls_per_query: int = DEFAULT_URLS_PER_QUERY,
    out_dir: str = "src/data/datasets/auto_data",
    max_capture_attempts: int = DEFAULT_MAX_CAPTURE_ATTEMPTS,
) -> None:
    """
    End-to-end Step 1:
    - generate OpenAI search queries from input types
    - retrieve candidate URLs via SerpApi
    - capture BrowserGym observation JSONs
    - save them with stable root names
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    queries = generate_search_queries(
        query_types_file=query_types_file,
        n_queries=n_search_queries,
    )

    # Save the generated search queries for debugging/reproducibility
    with open(out_path / "generated_search_queries.txt", "w", encoding="utf-8") as f:
        for q in queries:
            f.write(q + "\n")

    candidate_urls: List[str] = []
    for q in queries:
        try:
            urls = fetch_sites_from_serpapi(q, num_results=urls_per_query)
            candidate_urls.extend(urls)
        except Exception as e:
            print(f"[search-failed] query={q!r} error={e}", flush=True)

    candidate_urls = dedupe_preserve_order(candidate_urls)
    random.shuffle(candidate_urls)

    with open(out_path / "candidate_urls.txt", "w", encoding="utf-8") as f:
        for url in candidate_urls:
            f.write(url + "\n")

    dl = DownloaderEnv(DEFAULT_START_URL)
    saved = 0
    attempts = 0

    for url in candidate_urls:
        if saved >= n_websites or attempts >= max_capture_attempts:
            break

        attempts += 1
        print(f"[capture] {saved+1}/{n_websites} -> {url}", flush=True)

        try:
            obs = dl.obs_from_url(url)
            capture_id = save_observation(obs, source_query="serpapi_search", out_dir=out_path)
            print(f"[saved] {capture_id}.json", flush=True)
            saved += 1

        except Exception as e:
            print(f"[capture-failed] url={url!r} error={e}", flush=True)
            try:
                dl.page.close()
            except Exception:
                pass
            try:
                del dl
            except Exception:
                pass
            time.sleep(2)
            dl = DownloaderEnv(DEFAULT_START_URL)
            continue

    print(f"Done. Saved {saved} website JSON files to {out_path}")