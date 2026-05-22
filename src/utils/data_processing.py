from pathlib import Path
from typing import List, Dict

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_N_SEARCH_QUERIES = 12
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


def read_query_types_file(path: str) -> List[str]:
    """
    Reads a text file containing website/page types, one per line.
    Empty lines and comment lines starting with '#' are ignored.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    out = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)

    if not out:
        raise ValueError(f"No valid query types found in file: {path}")

    return out


def build_query_generation_prompt(query_types: List[str], n_queries: int) -> str:
    """
    Builds the OpenAI prompt used to generate concrete search queries.
    The model must return JSON so we can preserve the source type for each query.
    """
    joined_types = "\n".join(f"- {item}" for item in query_types)

    return f"""
You are helping build a dataset of websites for a web-navigation-agent project.

Below is a list of desired webpage or website types:
{joined_types}

Generate exactly {n_queries} diverse web search queries that are likely to retrieve real webpages matching these types.

Return STRICT JSON with this schema:
{{
  "queries": [
    {{
      "source_type": "<one of the provided webpage types>",
      "query": "<a realistic search-engine query>"
    }}
  ]
}}

Requirements:
- Return exactly {n_queries} query objects.
- Each query must be associated with one of the provided source types.
- "source_type" must exactly match one of the provided types.
- "query" must be a realistic search-engine query.
- Vary wording to improve diversity.
- Prefer queries that lead to publicly accessible webpages.
- Avoid queries likely to return only PDFs, app-store links, or login-blocked dashboards unless the type itself clearly calls for login pages.
- Keep the queries concise but specific.
- Output JSON only.
"""


def parse_query_generation_response(text: str, n_queries: int, query_types: List[str]) -> List[Dict[str, str]]:
    """
    Parses the JSON OpenAI response into a list of:
        {"source_type": ..., "query": ...}
    """
    import json
    import re

    text = text.strip()

    try:
        data = json.loads(text)
    except Exception:
        match = re.search(r"```json\s*(\{.*\})\s*```", text, re.DOTALL)
        if match:
            data = json.loads(match.group(1))
        else:
            match = re.search(r"(\{.*\})", text, re.DOTALL)
            if not match:
                raise ValueError("Could not parse JSON from OpenAI response.")
            data = json.loads(match.group(1))

    raw_queries = data.get("queries", [])
    if not isinstance(raw_queries, list):
        raise ValueError("OpenAI response JSON did not contain a valid 'queries' list.")

    valid_types = set(query_types)
    out: List[Dict[str, str]] = []
    seen = set()

    for item in raw_queries:
        if not isinstance(item, dict):
            continue

        source_type = str(item.get("source_type", "")).strip()
        query = str(item.get("query", "")).strip()

        if not source_type or not query:
            continue
        if source_type not in valid_types:
            continue

        key = (source_type, query)
        if key in seen:
            continue
        seen.add(key)

        out.append({
            "source_type": source_type,
            "query": query,
        })

    return out[:n_queries]


def generate_search_queries(
    query_types_file: str,
    n_queries: int = DEFAULT_N_SEARCH_QUERIES,
    model: str = DEFAULT_OPENAI_MODEL,
) -> List[Dict[str, str]]:
    """
    Uses the OpenAI API to generate search queries from a text file of website types.

    Returns:
        [
            {"source_type": "...", "query": "..."},
            ...
        ]
    """
    query_types = read_query_types_file(query_types_file)
    prompt = build_query_generation_prompt(query_types, n_queries)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model=model,
        temperature=0.8,
        messages=[
            {"role": "system", "content": "You are a precise assistant that returns valid JSON."},
            {"role": "user", "content": prompt},
        ],
    )

    text = response.choices[0].message.content or ""
    queries = parse_query_generation_response(text, n_queries, query_types)

    if not queries:
        raise ValueError("OpenAI returned no usable search queries.")

    return queries