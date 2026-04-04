from __future__ import annotations

import pytest

import ai
from ai import LLM, TextDelta


class _FakeContent:
    def __init__(self, chunks: list[str]) -> None:
        self._chunks = [chunk.encode("utf-8") for chunk in chunks]

    def __aiter__(self):
        self._iter = iter(self._chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class _FakeResponse:
    def __init__(self, chunks: list[str], *, status: int = 200, text_body: str = "") -> None:
        self.status = status
        self.content = _FakeContent(chunks)
        self._text_body = text_body

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def text(self) -> str:
        return self._text_body


class _FakeSession:
    def __init__(self, calls: list[dict], responses: list[_FakeResponse], timeout=None) -> None:
        self._calls = calls
        self._responses = responses
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def post(self, endpoint, headers=None, json=None):
        self._calls.append({
            "endpoint": endpoint,
            "headers": headers,
            "json": json,
        })
        return self._responses.pop(0)


def _patch_client_session(monkeypatch, calls: list[dict], responses: list[_FakeResponse]) -> None:
    def _factory(*args, **kwargs):
        return _FakeSession(calls, responses, timeout=kwargs.get("timeout"))

    monkeypatch.setattr(ai.aiohttp, "ClientSession", _factory)


@pytest.mark.asyncio
async def test_llama_cpp_cache_prompt_and_sticky_slot_reused(monkeypatch):
    calls: list[dict] = []
    responses = [
        _FakeResponse([
            'data: {"slot_id":3,"choices":[{"delta":{"content":"hi"}}]}\n',
            "data: [DONE]\n",
        ]),
        _FakeResponse([
            'data: {"choices":[{"delta":{"content":"again"}}]}\n',
            "data: [DONE]\n",
        ]),
    ]
    _patch_client_session(monkeypatch, calls, responses)

    llm = LLM(
        base_url="http://localhost:8080/v1",
        api_key="",
        model="test-model",
        llama_cpp_cache_prompt=True,
        llama_cpp_sticky_slots=True,
    )
    messages = [{"role": "user", "content": "hello"}]

    events = [event async for event in llm.stream(messages)]
    assert [event.text for event in events if isinstance(event, TextDelta)] == ["hi"]
    assert calls[0]["json"]["cache_prompt"] is True
    assert "id_slot" not in calls[0]["json"]
    assert llm._sticky_slot_id == 3

    events = [event async for event in llm.stream(messages)]
    assert [event.text for event in events if isinstance(event, TextDelta)] == ["again"]
    assert calls[1]["json"]["cache_prompt"] is True
    assert calls[1]["json"]["id_slot"] == 3
    assert calls[1]["json"]["slot_id"] == 3


@pytest.mark.asyncio
async def test_llama_cpp_slot_id_override_is_forwarded(monkeypatch):
    calls: list[dict] = []
    responses = [
        _FakeResponse([
            'data: {"choices":[{"delta":{"content":"ok"}}]}\n',
            "data: [DONE]\n",
        ]),
    ]
    _patch_client_session(monkeypatch, calls, responses)

    llm = LLM(
        base_url="http://localhost:8080/v1",
        api_key="",
        model="test-model",
        llama_cpp_cache_prompt=True,
        llama_cpp_sticky_slots=True,
        llama_cpp_slot_id=7,
    )

    _ = [event async for event in llm.stream([{"role": "user", "content": "hello"}])]

    assert calls[0]["json"]["cache_prompt"] is True
    assert calls[0]["json"]["id_slot"] == 7
    assert calls[0]["json"]["slot_id"] == 7
    assert llm._sticky_slot_id is None


@pytest.mark.asyncio
async def test_sticky_slot_can_be_captured_from_verbose_payload(monkeypatch):
    calls: list[dict] = []
    responses = [
        _FakeResponse([
            'data: {"__verbose":{"id_slot":5},"choices":[{"delta":{"content":"ok"}}]}\n',
            "data: [DONE]\n",
        ]),
    ]
    _patch_client_session(monkeypatch, calls, responses)

    llm = LLM(
        base_url="http://localhost:8080/v1",
        api_key="",
        model="test-model",
        llama_cpp_sticky_slots=True,
    )

    _ = [event async for event in llm.stream([{"role": "user", "content": "hello"}])]

    assert llm._sticky_slot_id == 5
