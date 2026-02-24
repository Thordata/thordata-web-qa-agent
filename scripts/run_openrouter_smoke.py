from __future__ import annotations

"""
Simple automation script to exercise Thordata Web QA Agent with OpenRouter.

It runs several real questions against the CLI using OpenRouter free models.
"""

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    env = os.environ.copy()
    # Ensure .env is loaded by the package (thordata env helpers handle this).
    # Here we only make sure the working dir is the project root.

    questions = [
        "What are the main use cases of Thordata for AI data pipelines?",
        "Compare Thordata's proxy network with traditional proxy providers.",
        "Summarize how Thordata helps with web scraping and SERP data collection.",
    ]

    # Based on verified free models; prefer the most reliable one first.
    models = [
        "stepfun/step-3.5-flash:free",
        "openrouter/free",
        "qwen/qwen3-coder:free",
    ]

    for q in questions:
        print("=" * 80)
        print(f"QUESTION: {q}")
        print("=" * 80)

        for model in models:
            cmd = [
                sys.executable,
                "-m",
                "thordata_web_qa_agent",
                "--question",
                q,
                "--backend",
                "openrouter",
                "--model",
                model,
            ]
            print(f"\n>>> Running with model={model}")
            try:
                result = subprocess.run(
                    cmd,
                    cwd=str(ROOT),
                    env=env,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=180,
                )
            except subprocess.TimeoutExpired:
                print("   !! Timeout, skipping this model.")
                continue

            if result.returncode != 0:
                print(f"   !! Exit code {result.returncode}")
                if result.stderr:
                    print(f"   STDERR (first 400 chars):\n{result.stderr[:400]}")
                continue

            out = result.stdout or ""
            print("   OK. STDOUT (first 800 chars):")
            print(out[:800])
            # If one model succeeds for this question, we can stop trying others.
            break


if __name__ == "__main__":
    main()

