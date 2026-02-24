"""
Backward-compatible entrypoint.

New implementation lives in the package:
  - `python -m thordata_web_qa_agent --question "..."`
  - `thordata-web-qa --question "..."`

Old usage remains supported:
  - `python web_qa_agent.py --question "..." [--offline] [--no-llm]`
"""

from __future__ import annotations

from thordata_web_qa_agent.cli import main


if __name__ == "__main__":
    main()
