import os
from pathlib import Path
from typing import List

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
    """
    joined_types = "\n".join(f"- {item}" for item in query_types)

    return f"""
You are helping build a dataset of websites for a web-navigation-agent project.

Below is a list of desired webpage or website types:
{joined_types}

Generate exactly {n_queries} diverse web search queries that are likely to retrieve real webpages matching these types.

Requirements:
- Return only a numbered list.
- Each item must be a realistic search-engine query.
- Vary wording to improve diversity.
- Prefer queries that lead to publicly accessible webpages.
- Avoid queries likely to return only PDFs, app-store links, or login-blocked dashboards unless the type itself clearly calls for login pages.
- Keep the queries concise but specific.
"""


def parse_query_generation_response(text: str, n_queries: int) -> List[str]:
    """
    Parses a numbered-list OpenAI response into a list of search queries.
    """
    queries = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # Accept simple numbered list items like "1. query text"
        if ". " in line and line[0].isdigit():
            parts = line.split(". ", 1)
            if len(parts) == 2:
                queries.append(parts[1].strip())

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            deduped.append(q)

    return deduped[:n_queries]


def generate_search_queries(
    query_types_file: str,
    n_queries: int = DEFAULT_N_SEARCH_QUERIES,
    model: str = DEFAULT_OPENAI_MODEL,
) -> List[str]:
    """
    Uses the OpenAI API to generate search queries from a text file of website types.
    """
    query_types = read_query_types_file(query_types_file)
    prompt = build_query_generation_prompt(query_types, n_queries)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model=model,
        temperature=0.8,
        messages=[
            {"role": "system", "content": "You are a precise assistant."},
            {"role": "user", "content": prompt},
        ],
    )

    text = response.choices[0].message.content or ""
    queries = parse_query_generation_response(text, n_queries)

    if not queries:
        raise ValueError("OpenAI returned no usable search queries.")

    return queries