from __future__ import annotations

import pytest

import ai
from ai import LLM, TextDelta, ToolCallAssembled


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


@pytest.mark.asyncio
async def test_responses_mode_chains_with_previous_response_id(monkeypatch):
    calls: list[dict] = []
    responses = [
        _FakeResponse([
            "event: response.created\n",
            'data: {"type":"response.created","response":{"id":"resp_1","status":"in_progress"}}\n',
            "\n",
            "event: response.output_text.delta\n",
            'data: {"type":"response.output_text.delta","item_id":"msg_1","delta":"Hello"}\n',
            "\n",
            "event: response.completed\n",
            'data: {"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Hello","annotations":[],"logprobs":[]}]}]}}\n',
            "\n",
            "data: [DONE]\n",
            "\n",
        ]),
        _FakeResponse([
            "event: response.created\n",
            'data: {"type":"response.created","response":{"id":"resp_2","status":"in_progress"}}\n',
            "\n",
            "event: response.output_text.delta\n",
            'data: {"type":"response.output_text.delta","item_id":"msg_2","delta":"Next"}\n',
            "\n",
            "event: response.completed\n",
            'data: {"type":"response.completed","response":{"id":"resp_2","status":"completed","output":[{"id":"msg_2","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Next","annotations":[],"logprobs":[]}]}]}}\n',
            "\n",
            "data: [DONE]\n",
            "\n",
        ]),
    ]
    _patch_client_session(monkeypatch, calls, responses)

    llm = LLM(
        base_url="http://localhost:8080/v1",
        api_key="",
        model="test-model",
        kind="responses",
    )

    first_messages = [{"role": "user", "content": "hello"}]
    first_events = [event async for event in llm.stream(first_messages)]
    assert [event.text for event in first_events if isinstance(event, TextDelta)] == ["Hello"]
    assert calls[0]["endpoint"].endswith("/responses")
    assert calls[0]["json"]["input"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "hello"}]}
    ]
    assert "previous_response_id" not in calls[0]["json"]
    assert llm._last_response_id == "resp_1"

    second_messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "follow up"},
    ]
    second_events = [event async for event in llm.stream(second_messages)]
    assert [event.text for event in second_events if isinstance(event, TextDelta)] == ["Next"]
    assert calls[1]["json"]["previous_response_id"] == "resp_1"
    assert calls[1]["json"]["input"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "follow up"}]}
    ]
    assert llm._last_response_id == "resp_2"


@pytest.mark.asyncio
async def test_responses_mode_drops_previous_response_id_when_history_diverges(monkeypatch):
    calls: list[dict] = []
    responses = [
        _FakeResponse([
            'data: {"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[{"type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Hello"}]}]}}\n',
            "\n",
            "data: [DONE]\n",
            "\n",
        ]),
        _FakeResponse([
            'data: {"type":"response.completed","response":{"id":"resp_2","status":"completed","output":[{"type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Other"}]}]}}\n',
            "\n",
            "data: [DONE]\n",
            "\n",
        ]),
    ]
    _patch_client_session(monkeypatch, calls, responses)

    llm = LLM(
        base_url="http://localhost:8080/v1",
        api_key="",
        model="test-model",
        kind="responses",
    )

    _ = [event async for event in llm.stream([{"role": "user", "content": "hello"}])]
    _ = [event async for event in llm.stream([{"role": "user", "content": "different"}])]

    assert "previous_response_id" not in calls[1]["json"]
    assert calls[1]["json"]["input"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "different"}]}
    ]


@pytest.mark.asyncio
async def test_responses_mode_disables_previous_response_id_for_llama_cpp(monkeypatch):
    calls: list[dict] = []
    responses = [
        _FakeResponse([
            'data: {"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[{"type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Hello"}]}]}}\n',
            "\n",
            "data: [DONE]\n",
            "\n",
        ]),
        _FakeResponse([
            "event: response.output_text.delta\n",
            'data: {"type":"response.output_text.delta","item_id":"msg_2","delta":"Next"}\n',
            "\n",
            'data: {"type":"response.completed","response":{"id":"resp_2","status":"completed","output":[{"type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Next"}]}]}}\n',
            "\n",
            "data: [DONE]\n",
            "\n",
        ]),
    ]
    _patch_client_session(monkeypatch, calls, responses)

    llm = LLM(
        base_url="http://localhost:8080/v1",
        api_key="",
        model="test-model",
        kind="responses",
        llama_cpp_cache_prompt=True,
        llama_cpp_sticky_slots=True,
    )

    _ = [event async for event in llm.stream([{"role": "user", "content": "hello"}])]

    second_messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "follow up"},
    ]
    second_events = [event async for event in llm.stream(second_messages)]

    assert [event.text for event in second_events if isinstance(event, TextDelta)] == ["Next"]
    assert "previous_response_id" not in calls[1]["json"]
    assert calls[1]["json"]["input"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "hello"}]},
        {"role": "assistant", "type": "message", "content": [{"type": "output_text", "text": "Hello"}]},
        {"role": "user", "content": [{"type": "input_text", "text": "follow up"}]},
    ]


@pytest.mark.asyncio
async def test_responses_mode_assembles_tool_calls_from_stream(monkeypatch):
    calls: list[dict] = []
    responses = [
        _FakeResponse([
            "event: response.output_item.added\n",
            'data: {"type":"response.output_item.added","item":{"type":"function_call","call_id":"fc_123","name":"shell","arguments":"","status":"in_progress"}}\n',
            "\n",
            "event: response.function_call_arguments.delta\n",
            'data: {"type":"response.function_call_arguments.delta","item_id":"fc_123","delta":"{\\"cmd\\": \\"dir\\"}"}\n',
            "\n",
            "event: response.output_item.done\n",
            'data: {"type":"response.output_item.done","item":{"type":"function_call","call_id":"fc_123","name":"shell","arguments":"{\\"cmd\\": \\"dir\\"}","status":"completed"}}\n',
            "\n",
            "event: response.completed\n",
            'data: {"type":"response.completed","response":{"id":"resp_tools","status":"completed","output":[{"type":"function_call","call_id":"fc_123","name":"shell","arguments":"{\\"cmd\\": \\"dir\\"}","status":"completed"}]}}\n',
            "\n",
            "data: [DONE]\n",
            "\n",
        ]),
    ]
    _patch_client_session(monkeypatch, calls, responses)

    llm = LLM(
        base_url="http://localhost:8080/v1",
        api_key="",
        model="test-model",
        kind="responses",
    )

    events = [event async for event in llm.stream([{"role": "user", "content": "run dir"}])]
    tool_events = [event for event in events if isinstance(event, ToolCallAssembled)]
    assert len(tool_events) == 1
    assert tool_events[0].call_id == "fc_123"
    assert tool_events[0].tool_name == "shell"
    assert tool_events[0].args == {"cmd": "dir"}


@pytest.mark.asyncio
async def test_responses_mode_accepts_native_responses_tool_defs(monkeypatch):
    calls: list[dict] = []
    responses = [
        _FakeResponse([
            'data: {"type":"response.completed","response":{"id":"resp_tool_passthrough","status":"completed","output":[{"type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"ok"}]}]}}\n',
            "\n",
            "data: [DONE]\n",
            "\n",
        ]),
    ]
    _patch_client_session(monkeypatch, calls, responses)

    llm = LLM(
        base_url="http://localhost:8080/v1",
        api_key="",
        model="test-model",
        kind="responses",
    )

    tools = [{
        "type": "function",
        "name": "lookup_weather",
        "description": "Lookup weather by city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }]
    _ = [event async for event in llm.stream([{"role": "user", "content": "weather?"}], tools=tools)]

    assert calls[0]["json"]["tools"] == tools
