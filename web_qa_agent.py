"""
Web Q&A Agent with Thordata (SERP + Universal Scraper + OpenAI)

Usage (live mode):
    python web_qa_agent.py --question "What is Thordata used for?"

Usage (offline mode, using cached docs in data/web_qa_sample.json):
    python web_qa_agent.py --question "..." --offline
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from thordata import ThordataClient, Engine

# Optional: OpenAI for LLM summarization
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # handled later


# -----------------------------
# Path & environment setup
# -----------------------------

# Assume this script lives at the project root
ROOT_DIR = Path(__file__).resolve().parent
ENV_PATH = ROOT_DIR / ".env"
DATA_DIR = ROOT_DIR / "data"
DOCS_CACHE_PATH = DATA_DIR / "web_qa_sample.json"

# Load .env if present
load_dotenv(ENV_PATH, override=True)


# -----------------------------
# Thordata client
# -----------------------------

SCRAPER_TOKEN = os.getenv("THORDATA_SCRAPER_TOKEN")
PUBLIC_TOKEN = os.getenv("THORDATA_PUBLIC_TOKEN")
PUBLIC_KEY = os.getenv("THORDATA_PUBLIC_KEY")

if not SCRAPER_TOKEN:
    raise RuntimeError(
        "THORDATA_SCRAPER_TOKEN is missing. "
        "Please configure your .env file at the project root."
    )

td_client = ThordataClient(
    scraper_token=SCRAPER_TOKEN,
    public_token=PUBLIC_TOKEN,
    public_key=PUBLIC_KEY,
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
    results = td_client.serp_search(
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
        html = td_client.universal_scrape(
            url=url,
            js_render=js_render,
            output_format="HTML",
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
        return (
            "OPENAI_API_KEY is missing. Please set it in your .env file to enable LLM calls."
        )

    client = OpenAI(api_key=OPENAI_API_KEY)

    # Build context from docs
    context_parts = []
    for idx, doc in enumerate(docs, start=1):
        context_parts.append(
            f"[Source {idx}] {doc.get('title')}\n"
            f"URL: {doc.get('url')}\n"
            f"Snippet: {doc.get('snippet')}\n"
            f"Content:\n{doc.get('content')}\n"
        )
    context_text = "\n\n".join(context_parts)

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

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
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
    args = parser.parse_args()

    use_live = not args.offline

    docs = get_docs_for_question(
        question=args.question,
        num_results=args.num_results,
        engine=Engine.GOOGLE,
        location=args.location,
        use_live_thordata=use_live,
    )

    # Show a small table of sources
    print()
    print(pd.DataFrame([{"title": d["title"], "url": d["url"]} for d in docs]))

    print(f"\nCollected {len(docs)} docs. Asking LLM...\n")
    answer = summarize_with_llm(args.question, docs)

    print("\n=== LLM Answer ===\n")
    print(answer)


if __name__ == "__main__":
    main()