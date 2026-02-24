from __future__ import annotations

"""
Scenario-based OpenRouter tests for Thordata Web QA Agent.

Covers:
- English technical question
- Chinese question
- Long compound question
- Edge case: very niche topic
"""

import os
import subprocess
import sys
from pathlib import Path


if sys.platform == "win32":
    try:
        import io

        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding="utf-8", errors="replace"
        )
    except Exception:
        pass


ROOT = Path(__file__).resolve().parents[1]


SCENARIOS = [
    {
        "name": "english_technical",
        "question": "Design a monitoring pipeline using Thordata to track competitor pricing for e-commerce products.",
    },
    {
        "name": "chinese_general",
        "question": "用通俗的语言说明 Thordata 的代理网络可以为大模型应用带来哪些实际好处？",
    },
    {
        "name": "long_compound",
        "question": (
            "Explain how Thordata can be integrated into an AI data pipeline that includes web scraping, "
            "SERP monitoring, data cleaning, and LLM-based summarization, and list potential failure points."
        ),
    },
    {
        "name": "edge_niche",
        "question": "Collect and summarize information about Thordata usage in academic research or scientific papers.",
    },
]


def main() -> None:
    env = os.environ.copy()

    model = "stepfun/step-3.5-flash:free"

    for s in SCENARIOS:
        print("=" * 80)
        print(f"SCENARIO: {s['name']}")
        print(f"QUESTION: {s['question']}")
        print("=" * 80)

        cmd = [
            sys.executable,
            "-m",
            "thordata_web_qa_agent",
            "--question",
            s["question"],
            "--backend",
            "openrouter",
            "--model",
            model,
            "--show-sources",
            "--verbose",
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(ROOT),
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=240,
            )
        except subprocess.TimeoutExpired:
            print("   !! Timeout in scenario:", s["name"])
            continue

        print(f"   Exit code: {result.returncode}")
        if result.stderr:
            print("   STDERR (first 400 chars):")
            print(result.stderr[:400])

        out = result.stdout or ""
        print("   STDOUT (first 1000 chars):")
        print(out[:1000])


if __name__ == "__main__":
    main()

