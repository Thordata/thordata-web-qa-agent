from __future__ import annotations

from thordata_web_qa_agent.agent import Document
from thordata_web_qa_agent.llm import summarize_with_llm


def test_summarize_with_llm_no_keys_returns_message() -> None:
    docs = [
        Document(
            url="https://example.com",
            title="Example",
            snippet="Snippet",
            content="Some content",
        )
    ]
    out = summarize_with_llm(
        question="test",
        docs=docs,
        model="gpt-4o-mini",
        backend="auto",
        openai_api_key=None,
        openrouter_api_key=None,
    )
    assert "No LLM credentials configured" in out
