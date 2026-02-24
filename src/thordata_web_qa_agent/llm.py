from __future__ import annotations

from typing import Any

from .agent import Document


def _build_context_blocks(docs: list[Document]) -> list[str]:
    parts: list[str] = []
    for i, d in enumerate(docs[:5], start=1):
        parts.append(
            f"[Source {i}] {d.title or ''}\n"
            f"URL: {d.url}\n"
            f"Snippet: {d.snippet or ''}\n"
            f"Content:\n{d.content}\n"
        )
    return parts


def _summarize_with_openai_client(
    *,
    question: str,
    docs: list[Document],
    model: str = "gpt-4o-mini",
    api_key: str,
    base_url: str | None = None,
) -> str:
    """
    Shared implementation for OpenAI-compatible HTTP APIs.

    This function imports OpenAI lazily so the base install can remain minimal.
    """
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return "OpenAI client is not installed. Install extras: pip install '.[llm]'"

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)  # type: ignore[arg-type]

    context_parts = _build_context_blocks(docs)

    system_prompt = (
        "You are a helpful web research assistant. "
        "Use ONLY the provided sources to answer the user's question. "
        "Respond in the same language as the question whenever possible. "
        "Include citations like [1], [2] that refer to the sources."
    )

    user_prompt = (
        f"Question:\n{question}\n\nSources:\n"
        + "\n\n".join(context_parts)
        + "\n\nAnswer concisely with citations, then list sources with URLs."
    )

    try:
        resp: Any = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
    except Exception as exc:  # noqa: BLE001
        # Return a concise, user-friendly error message instead of raising.
        return (
            f"LLM call failed (model={model!r}). "
            f"Please try a different free model or retry later.\nError: {exc}"
        )

    return str(getattr(resp.choices[0].message, "content", "") or "")


def summarize_with_llm(
    *,
    question: str,
    docs: list[Document],
    model: str,
    backend: str = "auto",
    openai_api_key: str | None = None,
    openrouter_api_key: str | None = None,
) -> str:
    """
    High-level LLM entrypoint with backend selection:
    - backend='openai'      → use OPENAI_API_KEY against api.openai.com
    - backend='openrouter'  → use OPENROUTER_API_KEY against openrouter.ai
    - backend='auto'        → prefer OpenRouter, then OpenAI, otherwise message.
    """
    backend_norm = (backend or "auto").lower()

    # Auto-detect
    if backend_norm == "auto":
        if openrouter_api_key:
            backend_norm = "openrouter"
        elif openai_api_key:
            backend_norm = "openai"
        else:
            return (
                "No LLM credentials configured. "
                "Set OPENROUTER_API_KEY or OPENAI_API_KEY, or pass --no-llm."
            )

    if backend_norm == "openrouter":
        if not openrouter_api_key:
            return (
                "OPENROUTER_API_KEY is missing. "
                "Set it in your environment or .env file."
            )
        return _summarize_with_openai_client(
            question=question,
            docs=docs,
            model=model,
            api_key=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    if backend_norm == "openai":
        if not openai_api_key:
            return (
                "OPENAI_API_KEY is missing. " "Set it in your environment or .env file."
            )
        return _summarize_with_openai_client(
            question=question,
            docs=docs,
            model=model,
            api_key=openai_api_key,
            base_url=None,
        )

    return (
        f"Unknown LLM backend: {backend}. "
        "Use one of: auto, openai, openrouter, or pass --no-llm."
    )


def summarize_with_openai(
    *,
    question: str,
    docs: list[Document],
    model: str = "gpt-4o-mini",
    api_key: str | None,
) -> str:
    """
    Backwards-compatible wrapper for OpenAI-only usage.
    """
    return summarize_with_llm(
        question=question,
        docs=docs,
        model=model,
        backend="openai",
        openai_api_key=api_key,
        openrouter_api_key=None,
    )


__all__ = ["summarize_with_llm", "summarize_with_openai"]
