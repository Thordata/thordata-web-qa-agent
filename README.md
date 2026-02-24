# Thordata Web Q&A Agent (Minimal)

Minimal Web Q&A CLI built on:

- Thordata **SERP API**
- Thordata **Universal Scrape** (Web Unlocker)
- Optional: **OpenAI / OpenRouter** summarization

This repository is intentionally small and easy to run. The project layout is inspired by `pypa/sampleproject` (src/ layout).

## Requirements

- Python 3.10+
- `THORDATA_SCRAPER_TOKEN` (required for SERP + Universal)
- Optional: `OPENAI_API_KEY` or `OPENROUTER_API_KEY` (only needed when LLM is enabled)

## Install

```bash
python -m pip install -e .
# dev tools + tests
python -m pip install -e ".[dev]"
# optional LLM
python -m pip install -e ".[llm]"
```

## Configuration

Copy `.env.example` to `.env` (never commit `.env`):

```bash
cp .env.example .env
```

The CLI loads `./.env` automatically (without overriding existing env vars).

## Usage

### LLM backend selection

- Auto (prefer OpenRouter, then OpenAI):

```bash
thordata-web-qa --question "What is Thordata used for?"
```

- Force OpenRouter:

```bash
thordata-web-qa --question "What is Thordata used for?" --backend openrouter --model mistralai/mistral-7b-instruct
```

- Force OpenAI:

```bash
thordata-web-qa --question "What is Thordata used for?" --backend openai --model gpt-4o-mini
```

Recommended default free model (already set as CLI default):

```bash
thordata-web-qa \
  --question "What is Thordata used for?" \
  --backend openrouter \
  --model stepfun/step-3.5-flash:free
```

You can also show collected sources and basic debug info:

```bash
thordata-web-qa \
  --question "What is Thordata used for?" \
  --backend openrouter \
  --show-sources \
  --verbose
```

### Live mode (collect + cache)

```bash
thordata-web-qa --question "What is Thordata used for?"
```

Cache is written to `data/web_qa_sample.json` by default.

### Offline mode (cache only)

```bash
thordata-web-qa --question "What is Thordata used for?" --offline
```

### No LLM (sources only)

```bash
thordata-web-qa --question "What is Thordata used for?" --no-llm
```

### Backward-compatible entrypoint

Old usage still works:

```bash
python web_qa_agent.py --question "What is Thordata used for?" --no-llm
```

## Test

```bash
pytest
```
