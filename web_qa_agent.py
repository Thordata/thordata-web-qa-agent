"""
Web Q&A Agent with Thordata (SERP + Universal Scraper + OpenAI)

Usage (live mode):
    python web_qa_agent.py --question "What is Thordata used for?"

Usage (offline mode, using cached docs in data/web_qa_sample.json):
    python web_qa_agent.py --question "..." --offline
"""

from __future__ import annotations

import argparse
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from thordata import (
    Engine,
    ThordataAPIError,
    ThordataAuthError,
    ThordataClient,
    ThordataConfigError,
    ThordataNetworkError,
    ThordataNotCollectedError,
    ThordataRateLimitError,
    ThordataTimeoutError,
)

# Optional: OpenAI for LLM summarization
try:
    from openai import OpenAI, RateLimitError  # type: ignore
except ImportError:
    OpenAI = None  # handled later
    RateLimitError = Exception  # type: ignore

# -----------------------------
# Path & environment setup
# -----------------------------

ROOT_DIR = Path(__file__).resolve().parent
ENV_PATH = ROOT_DIR / ".env"
DATA_DIR = ROOT_DIR / "data"
DOCS_CACHE_PATH = DATA_DIR / "web_qa_sample.json"

load_dotenv(ENV_PATH, override=True)

# -----------------------------
# Thordata client
# -----------------------------


@lru_cache
def get_thordata_client() -> ThordataClient:
    """
    Lazily create the Thordata client.

    This avoids requiring credentials at import time (important for offline mode and tests).
    """
    scraper_token = os.getenv("THORDATA_SCRAPER_TOKEN")
    public_token = os.getenv("THORDATA_PUBLIC_TOKEN")
    public_key = os.getenv("THORDATA_PUBLIC_KEY")

    if not scraper_token:
        raise ThordataConfigError(
            "THORDATA_SCRAPER_TOKEN is missing. Please configure your .env file."
        )

    return ThordataClient(
        scraper_token=scraper_token,
        public_token=public_token,
        public_key=public_key,
    )


# -----------------------------
# SERP + Universal helpers
# -----------------------------


def search_web_serp(
    query: str,
    num_results: int = 3,
    engine: Engine = Engine.GOOGLE,
    location: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Use Thordata SERP API to search the web and return a list of basic results.
    """
    extra_params: Dict[str, Any] = {}
    if location:
        extra_params["location"] = location

    print(f"Searching {engine.value} for: {query!r}")
    client = get_thordata_client()
    results = client.serp_search(
        query=query,
        engine=engine,
        num=num_results,
        **extra_params,
    )

    organic = results.get("organic") or []
    cleaned: List[Dict[str, Any]] = []
    for item in organic:
        cleaned.append(
            {
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet"),
            }
        )
    print(f"Got {len(cleaned)} organic results.")
    return cleaned


def clean_html_to_text(html: str) -> str:
    """
    Convert raw HTML into a cleaned plain-text representation.

    - Removes scripts, styles, navigation, footers, SVGs, iframes.
    - Collapses whitespace and drops empty lines.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove noisy elements
    for tag in soup(["script", "style", "nav", "footer", "svg", "iframe", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    clean_text = "\n".join(lines)
    return clean_text


def fetch_docs_from_web(
    query: str,
    num_results: int = 3,
    engine: Engine = Engine.GOOGLE,
    location: Optional[str] = None,
    js_render: bool = True,
    per_doc_max_chars: int = 4000,
) -> List[Dict[str, Any]]:
    """
    High-level function: search the web and fetch cleaned text for top results.

    Returns:
        List of documents: [{ "url", "title", "snippet", "content" }, ...]
    """
    serp_results = search_web_serp(
        query, num_results=num_results, engine=engine, location=location
    )

    docs: List[Dict[str, Any]] = []
    for idx, item in enumerate(serp_results, start=1):
        url = item.get("link")
        if not url:
            continue

        print(f"\n[{idx}/{len(serp_results)}] Fetching via Universal API: {url}")
        client = get_thordata_client()
        html = client.universal_scrape(
            url=url,
            js_render=js_render,
            output_format="html",
        )

        if not html or len(html) < 200:
            print("  Skipping: content too short or empty.")
            continue

        text = clean_html_to_text(html)
        if per_doc_max_chars and len(text) > per_doc_max_chars:
            text = text[:per_doc_max_chars]

        docs.append(
            {
                "url": url,
                "title": item.get("title"),
                "snippet": item.get("snippet"),
                "content": text,
            }
        )
        print(f"  Collected {len(text)} characters of cleaned text.")

    print(f"\nTotal documents collected: {len(docs)}")
    return docs


def get_docs_for_question(
    question: str,
    num_results: int = 3,
    engine: Engine = Engine.GOOGLE,
    location: Optional[str] = None,
    use_live_thordata: bool = True,
) -> List[Dict[str, Any]]:
    """
    Orchestrator: in live mode, fetch docs from web and cache them.
    In offline mode, load docs from the local cache.
    """
    if use_live_thordata:
        docs = fetch_docs_from_web(
            query=question,
            num_results=num_results,
            engine=engine,
            location=location,
            js_render=True,
            per_doc_max_chars=4000,
        )
        # Cache to local JSON
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with DOCS_CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump(docs, f, ensure_ascii=False, indent=2)
        print(f"\nCached docs to {DOCS_CACHE_PATH}")
    else:
        print(f"Loading docs from cache: {DOCS_CACHE_PATH}")
        if not DOCS_CACHE_PATH.is_file():
            raise FileNotFoundError(
                f"Cached docs not found at {DOCS_CACHE_PATH}. "
                "Run in live mode once to create them."
            )
        with DOCS_CACHE_PATH.open("r", encoding="utf-8") as f:
            docs = json.load(f)

    print(f"Loaded {len(docs)} documents.")
    return docs


# -----------------------------
# LLM summarization (OpenAI)
# -----------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Global limits to keep LLM context under control
MAX_CONTEXT_DOCS = 5
MAX_CONTEXT_CHARS = 15_000


def summarize_with_llm(
    question: str,
    docs: List[Dict[str, Any]],
    model: str = "gpt-4o-mini",
) -> str:
    """
    Ask an LLM to answer the question based on the provided documents.

    If OpenAI is not installed or API key is missing, a helpful message is returned.
    """
    if OpenAI is None:
        return (
            "LLM backend is not configured. Please install the 'openai' package:\n"
            "  pip install openai\n"
            "and set OPENAI_API_KEY in your .env file."
        )

    if not OPENAI_API_KEY:
        return "OPENAI_API_KEY is missing. Please set it in your .env file to enable LLM calls."

    # Limit number of docs to avoid huge prompts
    if len(docs) > MAX_CONTEXT_DOCS:
        docs_for_context = docs[:MAX_CONTEXT_DOCS]
    else:
        docs_for_context = docs

    client = OpenAI(api_key=OPENAI_API_KEY)

    # Build context from docs
    context_parts: List[str] = []
    for idx, doc in enumerate(docs_for_context, start=1):
        context_parts.append(
            f"[Source {idx}] {doc.get('title')}\n"
            f"URL: {doc.get('url')}\n"
            f"Snippet: {doc.get('snippet')}\n"
            f"Content:\n{doc.get('content')}\n"
        )
    context_text = "\n\n".join(context_parts)

    # Global char limit for safety
    if len(context_text) > MAX_CONTEXT_CHARS:
        context_text = context_text[:MAX_CONTEXT_CHARS]
        context_text += (
            f"\n\n[Context truncated to first {MAX_CONTEXT_CHARS} characters "
            "by web_qa_agent.py]"
        )

    system_prompt = (
        "You are a helpful web research assistant. "
        "Use ONLY the provided sources to answer the user's question. "
        "Include citations like [1], [2] that refer to the sources listed at the end."
    )

    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Sources:\n{context_text}\n\n"
        "Please provide a concise answer (in English or the question's language), "
        "with citations [1], [2], etc. Then list the sources with their URLs."
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
    except RateLimitError as e:
        return (
            "OpenAI returned 'insufficient_quota' for chat completions.\n"
            "Please check your OpenAI plan/billing before running this agent again.\n"
            f"Raw error: {e}"
        )

    answer = response.choices[0].message.content
    return answer


# -----------------------------
# CLI entrypoint
# -----------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Web Q&A Agent with Thordata (SERP + Universal + OpenAI)"
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Question to ask the web.",
    )
    parser.add_argument(
        "--num-results",
        type=int,
        default=3,
        help="Number of search results to fetch (default: 3).",
    )
    parser.add_argument(
        "--location",
        type=str,
        default=None,
        help="Optional location for SERP (e.g. 'United States').",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use cached docs from data/web_qa_sample.json instead of calling Thordata.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip the OpenAI call and only print collected documents/sources.",
    )
    args = parser.parse_args()

    use_live = not args.offline

    try:
        docs = get_docs_for_question(
            question=args.question,
            num_results=args.num_results,
            engine=Engine.GOOGLE,
            location=args.location,
            use_live_thordata=use_live,
        )
    except ThordataRateLimitError as e:
        print(
            "Thordata SERP/Universal API rate/quota issue (402/429).\n"
            "Please check your Thordata plan/balance:"
        )
        print(f"  {e}")
        return
    except ThordataAuthError as e:
        print("Thordata authentication/authorization error:")
        print(f"  {e}")
        return
    except ThordataAPIError as e:
        print("Thordata API returned an error while fetching docs:")
        print(f"  {e}")
        return
    except FileNotFoundError as e:
        # Offline mode but cache file missing
        print(str(e))
        return
    except ThordataNotCollectedError as e:
        print(
            "Thordata returned 'Not collected' (code=300). "
            "Consider retrying or broadening the query."
        )
        print(f"  {e}")
        return
    except ThordataTimeoutError as e:
        print("Thordata request timed out.")
        print(f"  {e}")
        return
    except ThordataNetworkError as e:
        print("Network error while calling Thordata.")
        print(f"  {e}")
        return
    except ThordataConfigError as e:
        print("Thordata configuration error:")
        print(f"  {e}")
        return

    if not docs:
        print("No documents collected. Try a broader question or check Thordata logs.")
        return

    if args.no_llm:
        print("\nSkipping LLM. Collected documents:")
        for i, d in enumerate(docs, start=1):
            print(f"[{i}] {d.get('title')}\n    {d.get('url')}\n")
        return

    # Show a small table of sources
    print()
    print(pd.DataFrame([{"title": d["title"], "url": d["url"]} for d in docs]))

    print(f"\nCollected {len(docs)} docs. Asking LLM...\n")
    answer = summarize_with_llm(args.question, docs)

    print("\n=== LLM Answer ===\n")
    print(answer)


if __name__ == "__main__":
    main()
