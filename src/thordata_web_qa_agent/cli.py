from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from thordata import Engine

from .agent import (
    ThordataAPIError,
    ThordataAuthError,
    ThordataConfigError,
    ThordataNetworkError,
    ThordataNotCollectedError,
    ThordataRateLimitError,
    ThordataTimeoutError,
    collect_documents,
    default_cache_path,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Thordata Web Q&A (minimal CLI)")
    p.add_argument("--question", type=str, required=True, help="Question to ask.")
    p.add_argument("--num-results", type=int, default=3, help="SERP results to fetch.")
    p.add_argument("--offline", action="store_true", help="Use cached docs only.")
    p.add_argument("--no-llm", action="store_true", help="Skip LLM summarization.")
    p.add_argument(
        "--cache",
        type=str,
        default=None,
        help="Cache file path (default: data/web_qa_sample.json).",
    )
    p.add_argument(
        "--engine",
        type=str,
        default="google",
        choices=["google", "bing"],
        help="Search engine (default: google).",
    )
    p.add_argument("--country", type=str, default=None, help="Country code (e.g. us).")
    p.add_argument("--language", type=str, default=None, help="Language code (e.g. en).")
    p.add_argument(
        "--location",
        type=str,
        default=None,
        help="Legacy SERP location (passed through to SDK as extra param).",
    )
    p.add_argument(
        "--model",
        type=str,
        default="stepfun/step-3.5-flash:free",
        help="LLM model name (depends on backend, only used when LLM is enabled).",
    )
    p.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "openai", "openrouter"],
        help="LLM backend: auto (default), openai, or openrouter.",
    )
    p.add_argument(
        "--per-doc-max-chars",
        type=int,
        default=4000,
        help="Max chars per document after cleaning (default: 4000).",
    )
    p.add_argument(
        "--show-sources",
        action="store_true",
        help="Print collected source titles and URLs before the answer.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose mode: echo backend/model and basic debug info.",
    )
    return p


def _engine_from_str(v: str) -> Engine:
    return Engine.GOOGLE if v.lower() == "google" else Engine.BING


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    # Make stdout/stderr UTF-8 friendly on Windows to avoid UnicodeEncodeError.
    if os.name == "nt":
        try:
            import io

            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace"
            )
            sys.stderr = io.TextIOWrapper(
                sys.stderr.buffer, encoding="utf-8", errors="replace"
            )
        except Exception:
            # Best-effort only; ignore if wrapping fails.
            pass

    cache_path = Path(args.cache) if args.cache else default_cache_path()
    engine = _engine_from_str(args.engine)

    try:
        docs = collect_documents(
            question=args.question,
            offline=bool(args.offline),
            cache_path=cache_path,
            num_results=int(args.num_results),
            engine=engine,
            country=args.country,
            language=args.language,
            location=args.location,
            js_render=True,
            per_doc_max_chars=int(args.per_doc_max_chars),
        )
    except ThordataRateLimitError as e:
        print("Thordata rate/quota issue (402/429).")
        print(str(e))
        raise SystemExit(2) from e
    except ThordataAuthError as e:
        print("Thordata auth error.")
        print(str(e))
        raise SystemExit(2) from e
    except (ThordataAPIError, ThordataNetworkError, ThordataTimeoutError) as e:
        print("Thordata request failed.")
        print(str(e))
        raise SystemExit(2) from e
    except ThordataNotCollectedError as e:
        print("Thordata returned 'Not collected' (code=300).")
        print(str(e))
        raise SystemExit(2) from e
    except ThordataConfigError as e:
        print("Configuration error.")
        print(str(e))
        raise SystemExit(2) from e
    except FileNotFoundError as e:
        print(str(e))
        raise SystemExit(2) from e

    if not docs:
        print("No documents collected.")
        raise SystemExit(0)

    if args.verbose:
        print(f"Using backend={args.backend}, model={args.model}")
        print(f"Collected {len(docs)} documents.")

    if args.show_sources:
        print("Sources:")
        for i, d in enumerate(docs, start=1):
            print(f"[{i}] {d.title or ''}\n    {d.url}")
        print()

    if args.no_llm:
        print("Skipping LLM. Collected sources:")
        for i, d in enumerate(docs, start=1):
            print(f"[{i}] {d.title or ''}\n    {d.url}")
        return

    from .llm import summarize_with_llm

    answer = summarize_with_llm(
        question=args.question,
        docs=docs,
        model=str(args.model),
        backend=str(args.backend),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    sys.stdout.write(answer.strip() + "\n")


__all__ = ["main", "build_parser"]

