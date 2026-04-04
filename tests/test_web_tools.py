"""
tests/test_web_tools.py

Tests for the web module's direct browsing/scraping helpers and tool wiring.

Run with:
    python -m pytest tests/test_web_tools.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.web import __main__ as web_module


class _MockConfig:
    def __init__(self, ws_path: str, extra: dict | None = None):
        self.workspace = type("WS", (), {"path": ws_path})()
        self.extra = extra or {}


class _MockToolHandler:
    def __init__(self):
        self.tools: dict[str, object] = {}
        self.always_on: dict[str, bool] = {}

    def register_tool(self, func, always_on=False):
        self.tools[func.__name__] = func
        self.always_on[func.__name__] = always_on


class _MockContext:
    def __init__(self):
        self.prompts: dict[str, tuple[object, object]] = {}

    def register_prompt(self, pid, provider, *, role="system", priority=0):
        self.prompts[pid] = (
            type("PromptSlot", (), {"pid": pid, "role": role, "priority": priority})(),
            provider,
        )


class _MockAgent:
    def __init__(self, ws_path: str, extra: dict | None = None):
        self.config = _MockConfig(ws_path, extra=extra)
        self.tool_handler = _MockToolHandler()
        self.context = _MockContext()
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1


@pytest.fixture
def agent(tmp_path: Path) -> _MockAgent:
    return _MockAgent(str(tmp_path))


def test_register_exposes_browse_url_as_always_on(agent: _MockAgent):
    web_module.register(agent)

    assert "browse_url" in agent.tool_handler.tools
    assert agent.tool_handler.always_on["browse_url"] is True


def test_register_adds_web_prompt(agent: _MockAgent):
    web_module.register(agent)

    slot, provider = agent.context.prompts["web_tools"]
    prompt = provider(None)

    assert slot.role == "system"
    assert slot.priority == 12
    assert "Use browse_url when you already have a specific http/https URL" in prompt
    assert "Do not use shell with curl" in prompt


def test_html_to_text_strips_non_visible_tags():
    html = """
    <html>
      <head>
        <title>Ignore me</title>
        <style>body { color: red; }</style>
      </head>
      <body>
        <h1>Hello</h1>
        <p>World</p>
        <script>alert("x")</script>
      </body>
    </html>
    """

    assert web_module._html_to_text(html) == "Hello\n\nWorld"


def test_parse_duckduckgo_results_decodes_redirect_links():
    html = """
    <div class="result results_links results_links_deep web-result">
      <div class="links_main links_deep result__body">
        <h2 class="result__title">
          <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fdocs">
            Example Docs
          </a>
        </h2>
        <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fdocs">
          Useful snippet text.
        </a>
      </div>
    </div>
    """

    results = web_module._parse_duckduckgo_results(html, max_results=5)

    assert results == [{
        "title": "Example Docs",
        "href": "https://example.com/docs",
        "body": "Useful snippet text.",
    }]


@pytest.mark.asyncio
async def test_browse_url_uses_http_helper_by_default(agent: _MockAgent, monkeypatch):
    web_module.register(agent)

    async def fake_browse_with_http(
        url: str,
        mode: str,
        *,
        max_chars: int,
        max_bytes: int,
        timeout_ms: int,
        user_agent: str,
        ignored_tags: list[str] | None = None,
    ) -> dict:
        assert url == "https://example.com/docs"
        assert mode == "text"
        assert max_chars == 20000
        assert max_bytes == 2000000
        assert timeout_ms == 30000
        assert user_agent == "TinyCTX/1.1"
        assert ignored_tags == ["script", "style"]
        return {
            "url": url,
            "final_url": url,
            "status_code": 200,
            "content_type": "text/html; charset=utf-8",
            "mode": mode,
            "rendered": False,
            "truncated": False,
            "bytes": 1234,
            "title": "Example Docs",
            "content": "Hello world",
        }

    monkeypatch.setattr(web_module, "_browse_with_http", fake_browse_with_http)

    result = await agent.tool_handler.tools["browse_url"]("https://example.com/docs")
    payload = json.loads(result)

    assert payload["rendered"] is False
    assert payload["title"] == "Example Docs"
    assert payload["content"] == "Hello world"


@pytest.mark.asyncio
async def test_browse_url_can_render_js(agent: _MockAgent, monkeypatch):
    web_module.register(agent)

    async def fake_browse_with_browser(
        agent_obj,
        url: str,
        mode: str,
        *,
        max_chars: int,
        timeout_ms: int,
        ignored_tags: list[str] | None = None,
    ) -> dict:
        assert agent_obj is agent
        assert url == "https://example.com/app"
        assert mode == "html"
        assert max_chars == 250
        assert timeout_ms == 30000
        assert ignored_tags == ["script", "style"]
        return {
            "url": url,
            "final_url": url,
            "status_code": 200,
            "content_type": "text/html",
            "mode": mode,
            "rendered": True,
            "truncated": False,
            "bytes": 999,
            "title": "Rendered App",
            "content": "<html><body>hydrated</body></html>",
        }

    monkeypatch.setattr(web_module, "_browse_with_browser", fake_browse_with_browser)

    result = await agent.tool_handler.tools["browse_url"](
        "https://example.com/app",
        mode="html",
        render_js=True,
        max_chars=250,
    )
    payload = json.loads(result)

    assert payload["rendered"] is True
    assert payload["mode"] == "html"
    assert "hydrated" in payload["content"]


@pytest.mark.asyncio
async def test_browse_url_rejects_invalid_inputs(agent: _MockAgent):
    web_module.register(agent)
    browse_url = agent.tool_handler.tools["browse_url"]

    bad_scheme = await browse_url("file:///etc/passwd")
    bad_mode = await browse_url("https://example.com", mode="markdown")
    bad_max_chars = await browse_url("https://example.com", max_chars=0)

    assert "http:// or https://" in bad_scheme
    assert "mode must be 'text' or 'html'" in bad_mode
    assert "greater than 0" in bad_max_chars


@pytest.mark.asyncio
async def test_runtime_web_settings_override_defaults(tmp_path: Path, monkeypatch):
    agent = _MockAgent(
        str(tmp_path),
        extra={
            "web": {
                "browse_max_bytes": 4096,
                "browse_max_chars": 120,
                "browse_user_agent": "TinyCTX-Test/1.0",
            }
        },
    )
    web_module.register(agent)

    async def fake_browse_with_http(
        url: str,
        mode: str,
        *,
        max_chars: int,
        max_bytes: int,
        timeout_ms: int,
        user_agent: str,
        ignored_tags: list[str] | None = None,
    ) -> dict:
        assert max_chars == 120
        assert max_bytes == 4096
        assert user_agent == "TinyCTX-Test/1.0"
        return {
            "url": url,
            "final_url": url,
            "status_code": 200,
            "content_type": "text/plain",
            "mode": mode,
            "rendered": False,
            "truncated": False,
            "bytes": 5,
            "title": None,
            "content": "hello",
        }

    monkeypatch.setattr(web_module, "_browse_with_http", fake_browse_with_http)

    result = await agent.tool_handler.tools["browse_url"]("https://example.com")
    payload = json.loads(result)
    assert payload["content"] == "hello"


@pytest.mark.asyncio
async def test_web_search_falls_back_when_ddgs_missing(agent: _MockAgent, monkeypatch):
    web_module.register(agent)

    async def fake_search_with_duckduckgo_html(query: str, *, num_results: int, user_agent: str):
        assert query == "moltbook twitter verification signup"
        assert num_results == 3
        assert user_agent == "TinyCTX/1.1"
        return [{
            "title": "Moltbook",
            "href": "https://www.moltbook.com/",
            "body": "Agent internet front page.",
        }]

    monkeypatch.setattr(web_module, "_search_with_duckduckgo_html", fake_search_with_duckduckgo_html)
    monkeypatch.setitem(sys.modules, "ddgs", None)

    result = await agent.tool_handler.tools["web_search"](
        "moltbook twitter verification signup",
        num_results=3,
    )

    assert "Search results for 'moltbook twitter verification signup':" in result
    assert "https://www.moltbook.com/" in result
