import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import parse_qs

from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Request, Response


def test_web_qa_agent_runs_with_mocked_thordata_no_llm(
    httpserver: HTTPServer, tmp_path: Path
) -> None:
    """
    E2E test:
    - Mock SERP endpoint (/request)
    - Mock Universal endpoint (/request)
    - Run CLI with --no-llm so no OpenAI call is needed
    """

    def handler(request: Request) -> Response:
        body = request.get_data(as_text=True) or ""
        form = parse_qs(body)

        # SERP payload includes "engine"
        if "engine" in form:
            payload = {
                "code": 200,
                "organic": [
                    {
                        "title": "Example Result",
                        "link": "https://example.com",
                        "snippet": "Example snippet",
                    }
                ],
            }
            return Response(
                json.dumps(payload), status=200, content_type="application/json"
            )

        # Universal payload includes "type"
        if "type" in form:
            long_html = "<html><body>" + ("Hello " * 60) + "</body></html>"
            payload = {"code": 200, "html": long_html}
            return Response(
                json.dumps(payload), status=200, content_type="application/json"
            )

        return Response(
            json.dumps({"code": 400, "msg": "Bad request"}),
            status=200,
            content_type="application/json",
        )

    httpserver.expect_request("/request", method="POST").respond_with_handler(handler)
    base_url = httpserver.url_for("/").rstrip("/").replace("localhost", "127.0.0.1")

    env = os.environ.copy()
    env["THORDATA_SCRAPER_TOKEN"] = "dummy"
    env["THORDATA_SCRAPERAPI_BASE_URL"] = base_url
    env["THORDATA_UNIVERSALAPI_BASE_URL"] = base_url
    env["THORDATA_WEB_QA_CACHE_DIR"] = str(tmp_path)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["NO_PROXY"] = "127.0.0.1,localhost"
    env["no_proxy"] = env["NO_PROXY"]

    result = subprocess.run(
        [sys.executable, "web_qa_agent.py", "--question", "test question", "--no-llm"],
        cwd=str(Path(__file__).resolve().parents[1]),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )

    assert result.returncode == 0, (result.stdout or "") + "\n" + (result.stderr or "")
    out = (result.stdout or "").lower()
    assert "skipping llm" in out
