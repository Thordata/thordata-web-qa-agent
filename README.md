# Thordata Web Q&A Agent

A minimal **Web Question‑Answering Agent** built on top of:

- Thordata **SERP API** (Google search)
- Thordata **Universal Scraper API** (html scraping + JS rendering)
- **OpenAI** chat models for final answers

The agent:

1. Takes a natural‑language question.
2. Uses Thordata SERP API to search the web.
3. Uses Thordata Universal Scraper API to fetch and clean page content.
4. Calls an OpenAI model (e.g. `gpt-4o-mini`) to answer the question, including citations and source URLs.

Supports both **live mode** (call real APIs) and **offline mode** (reuse cached documents).

---

## 🧩 Requirements

- Python 3.10+ (3.11 recommended)
- A Thordata account and API credentials:
  - `THORDATA_SCRAPER_TOKEN`
  - `THORDATA_PUBLIC_TOKEN`
  - `THORDATA_PUBLIC_KEY`
- An OpenAI API key (`OPENAI_API_KEY`) for LLM answers

---

## 📦 Installation

Clone this repository and create a virtual environment:

```bash
git clone https://github.com/Thordata/thordata-web-qa-agent.git
cd thordata-web-qa-agent

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
# source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 🔐 Configuration

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```env
THORDATA_SCRAPER_TOKEN=your_thordata_scraper_token
THORDATA_PUBLIC_TOKEN=your_thordata_public_token
THORDATA_PUBLIC_KEY=your_thordata_public_key

OPENAI_API_KEY=sk-...
```

The script loads `.env` automatically from the project root.

---

## 🚀 Usage

### Live mode (call Thordata + OpenAI)

This will:

1. Use Thordata SERP API to search Google.
2. Use Universal Scraper API to fetch and clean the top results.
3. Cache the cleaned documents under `data/web_qa_sample.json`.
4. Ask OpenAI to answer the question based on those documents.

```bash
python web_qa_agent.py \
  --question "What are the main use cases of Thordata for AI data pipelines?"
```

**Sample output:**

```
Searching google for: 'What are the main use cases of Thordata for AI data pipelines?'
Got 3 organic results.

[1/3] Fetching via Universal API: https://www.thordata.com/blog/proxies/thordata-review
  Collected 4000 characters of cleaned text.

[2/3] Fetching via Universal API: https://www.mexc.co/news/196212
  Collected 4000 characters of cleaned text.

Total documents collected: 2
Cached docs to .../data/web_qa_sample.json
Loaded 2 documents.

Collected 2 docs. Asking LLM...

=== LLM Answer ===
...
```

### Offline mode (reuse cached docs only)

Once you have run the script in live mode at least once (so that `data/web_qa_sample.json` exists), you can switch to offline mode to avoid calling Thordata SERP / Universal again:

```bash
python web_qa_agent.py \
  --question "What is Thordata used for?" \
  --offline
```

**Offline mode:**
- Reads documents from `data/web_qa_sample.json`
- Still calls OpenAI for the final answer

---

## ⚙️ How it works

Internally, `web_qa_agent.py` is composed of:

### `search_web_serp(...)`
Thin wrapper around `ThordataClient.serp_search` (Google by default) that extracts `{title, link, snippet}`.

### `fetch_docs_from_web(...)`
For each SERP result, calls `ThordataClient.universal_scrape` to fetch html, then cleans it into plain text using BeautifulSoup, truncating to a safe length per document.

### `get_docs_for_question(...)`
Orchestrator that either:
- Uses live SERP + Universal and writes `data/web_qa_sample.json`, or
- Loads cached docs from `data/web_qa_sample.json` in offline mode.

### `summarize_with_llm(...)`
Builds a prompt with all sources and calls the OpenAI Chat Completions API to generate an answer with citations and a source list.

---

## 📂 Project structure

```
thordata-web-qa-agent/
├── web_qa_agent.py         # main CLI script
├── requirements.txt        # Python dependencies
├── .env.example            # example configuration (copy to .env)
├── .gitignore
└── data/
    └── web_qa_sample.json  # cached documents (created at runtime, git-ignored)
```

---

## 📝 License

This project is provided as an example and is licensed under the MIT License. See the LICENSE file for details.
