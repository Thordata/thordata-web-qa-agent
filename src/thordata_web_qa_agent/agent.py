from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from thordata import Engine, ThordataClient, load_env_file
from thordata.exceptions import (
    ThordataAPIError,
    ThordataAuthError,
    ThordataConfigError,
    ThordataNetworkError,
    ThordataNotCollectedError,
    ThordataRateLimitError,
    ThordataTimeoutError,
)


@dataclass(frozen=True)
class Document:
    url: str
    title: str | None
    snippet: str | None
    content: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "content": self.content,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Document:
        return Document(
            url=str(d.get("url") or ""),
            title=d.get("title"),
            snippet=d.get("snippet"),
            content=str(d.get("content") or ""),
        )


def load_local_env() -> None:
    """
    Load ./.env if present, without overriding existing environment variables.

    This uses the SDK helper (dependency-free) instead of python-dotenv.
    """
    load_env_file(".env", override=False)


def _get_required_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise ThordataConfigError(f"Missing required environment variable: {name}")
    return v


def create_client() -> ThordataClient:
    """
    Create a ThordataClient from environment variables.

    Notes:
    - For SERP + Universal, only THORDATA_SCRAPER_TOKEN is required.
    - Public token/key are only required for Web Scraper Tasks and management APIs.
    """
    scraper_token = _get_required_env("THORDATA_SCRAPER_TOKEN")
    public_token = os.getenv("THORDATA_PUBLIC_TOKEN")
    public_key = os.getenv("THORDATA_PUBLIC_KEY")

    return ThordataClient(
        scraper_token=scraper_token,
        public_token=public_token,
        public_key=public_key,
    )


def search_web(
    client: ThordataClient,
    *,
    query: str,
    num_results: int = 3,
    engine: Engine = Engine.GOOGLE,
    country: str | None = None,
    language: str | None = None,
    location: str | None = None,
) -> list[dict[str, Any]]:
    results = client.serp_search(
        query=query,
        engine=engine,
        num=num_results,
        country=country,
        language=language,
        output_format="json",
        location=location,
    )
    organic = results.get("organic") or []
    cleaned: list[dict[str, Any]] = []
    for item in organic:
        cleaned.append(
            {
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet"),
            }
        )
    return cleaned


def fetch_documents(
    client: ThordataClient,
    *,
    query: str,
    num_results: int = 3,
    engine: Engine = Engine.GOOGLE,
    country: str | None = None,
    language: str | None = None,
    location: str | None = None,
    js_render: bool = True,
    per_doc_max_chars: int = 4000,
) -> list[Document]:
    serp = search_web(
        client,
        query=query,
        num_results=num_results,
        engine=engine,
        country=country,
        language=language,
        location=location,
    )

    docs: list[Document] = []
    for item in serp:
        url = item.get("link") or ""
        if not url:
            continue

        text = client.universal_scrape_markdown(
            url=url,
            js_render=js_render,
            max_chars=max(per_doc_max_chars, 1),
            country=country,
        )

        text = text.strip()
        # Allow shorter pages while still filtering obvious junk.
        if len(text) < 30:
            continue

        docs.append(
            Document(
                url=url,
                title=item.get("title"),
                snippet=item.get("snippet"),
                content=text,
            )
        )

    return docs


def default_cache_path() -> Path:
    """
    Default cache path:
    - If THORDATA_WEB_QA_CACHE_DIR is set, store cache under that directory.
    - Otherwise use ./data/web_qa_sample.json (relative to current working dir).
    """
    override_dir = os.getenv("THORDATA_WEB_QA_CACHE_DIR")
    if override_dir:
        return Path(override_dir) / "web_qa_sample.json"
    return Path("data") / "web_qa_sample.json"


def save_cache(path: Path, docs: list[Document]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [d.to_dict() for d in docs]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_cache(path: Path) -> list[Document]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Cache file not found: {path}. Run once without --offline to create it."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Invalid cache format: expected a list of documents")
    return [Document.from_dict(x) for x in raw if isinstance(x, dict)]


def collect_documents(
    *,
    question: str,
    offline: bool,
    cache_path: Path,
    num_results: int = 3,
    engine: Engine = Engine.GOOGLE,
    country: str | None = None,
    language: str | None = None,
    location: str | None = None,
    js_render: bool = True,
    per_doc_max_chars: int = 4000,
) -> list[Document]:
    if offline:
        return load_cache(cache_path)

    load_local_env()

    # Simple heuristic: if the question contains non-ASCII characters and
    # no language was provided, default to Chinese search parameters.
    if language is None and any(ord(ch) > 127 for ch in question):
        language = "zh"
        if country is None:
            country = "cn"

    client = create_client()

    docs = fetch_documents(
        client,
        query=question,
        num_results=num_results,
        engine=engine,
        country=country,
        language=language,
        location=location,
        js_render=js_render,
        per_doc_max_chars=per_doc_max_chars,
    )
    save_cache(cache_path, docs)
    return docs


__all__ = [
    "Document",
    "ThordataAPIError",
    "ThordataAuthError",
    "ThordataConfigError",
    "ThordataNetworkError",
    "ThordataNotCollectedError",
    "ThordataRateLimitError",
    "ThordataTimeoutError",
    "collect_documents",
    "default_cache_path",
    "load_local_env",
]
