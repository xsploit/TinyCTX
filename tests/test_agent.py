"""
tests/test_agent.py

Tests for AgentLoop — the 6-stage execution loop.

The LLM is fully stubbed via async generator functions so no network calls
are made. Modules are not loaded (MODULES_DIR is patched to a non-existent
path). Each make_agent() call gets a completely fresh AgentLoop with its own
unique session key, so tests never bleed into each other.

Run with:
    python -m pytest tests/
"""
import asyncio
import json
import time
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from contracts import (
    SessionKey, UserIdentity, InboundMessage,
    AgentTextChunk, AgentTextFinal, AgentError,
    Platform, ContentType, ToolCall, ToolResult,
)
from context import HistoryEntry
from ai import TextDelta, ToolCallAssembled, LLMError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path):
    cfg = MagicMock()
    cfg.models = {"primary": MagicMock(), "fast": MagicMock()}
    cfg.llm.primary = "primary"
    cfg.llm.fallback = []
    cfg.llm.fallback_on.any_error = False
    cfg.llm.fallback_on.http_codes = [429, 500]
    cfg.context = 4096
    cfg.max_tool_cycles = 5
    cfg.memory.workspace_path = str(tmp_path)
    return cfg


def _make_msg(text="hello", session_key=None):
    sk = session_key or SessionKey.dm("u1")
    return InboundMessage(
        session_key=sk,
        author=UserIdentity(platform=Platform.CLI, user_id="u1", username="alice"),
        content_type=ContentType.TEXT,
        text=text,
        message_id="msg-1",
        timestamp=0.0,
    )


async def _collect(agent, msg):
    """Run the agent and collect all reply chunks (partials + final)."""
    chunks = []
    async for chunk in agent.run(msg):
        chunks.append(chunk)
    return chunks


def _full_text(chunks):
    """
    Reassemble the full reply text from a list of agent events.
    AgentTextChunk events are streaming tokens; AgentTextFinal closes the
    stream (text may be empty sentinel or full text on non-streaming path);
    AgentError carries an error message.
    """
    return "".join(c.text if hasattr(c, 'text') else c.message for c in chunks)


def _text_stream(*texts):
    """Return an async generator that yields TextDeltas then stops."""
    async def _gen(messages, tools=None):
        for t in texts:
            yield TextDelta(text=t)
    return _gen


def _error_stream(message):
    """Return an async generator that yields a single LLMError."""
    async def _gen(messages, tools=None):
        yield LLMError(message=message)
    return _gen


# ---------------------------------------------------------------------------
# Fixture: AgentLoop with LLM and modules patched
# ---------------------------------------------------------------------------

@pytest.fixture
def make_agent(tmp_path):
    """
    Factory fixture — call make_agent(stream_fn) to get a fresh AgentLoop.

    Every call gets a unique session key (test-user-1, test-user-2, ...) so
    _restore_history never picks up state from a prior call, and context
    never bleeds between tests.

    Session files are written to tmp_path/sessions/ so persistence tests
    can find them reliably without touching the real workspace.
    """
    counter = {"n": 0}

    def _factory(stream_fn=None, fallback_stream_fn=None, fallback_names=None):
        from agent import AgentLoop

        # Unique key per call — prevents _restore_history cross-contamination
        counter["n"] += 1
        unique_key = SessionKey.dm(f"test-user-{counter['n']}")

        cfg = _make_config(tmp_path)

        if fallback_names:
            cfg.llm.fallback = fallback_names
            cfg.llm.fallback_on.any_error = True

        primary_llm = MagicMock()
        primary_llm.stream = stream_fn or _text_stream("default reply")

        fallback_llm = MagicMock()
        fallback_llm.stream = fallback_stream_fn or _text_stream("fallback reply")

        # Patch MODULES_DIR so no real modules are loaded, and _build_llm so
        # no real API clients are constructed during __init__.
        with patch("agent.MODULES_DIR", Path("/nonexistent")):
            with patch("agent._build_llm", return_value=primary_llm):
                agent = AgentLoop(session_key=unique_key, config=cfg)

        # Wire controlled stubs into the model pool after construction.
        agent._models["primary"] = primary_llm
        if fallback_names:
            for name in fallback_names:
                agent._models[name] = fallback_llm

        # Redirect session file writes to tmp_path so persistence tests work.
        # The real _flush_history resolves paths from Path("sessions/") relative
        # to the CWD (the repo root during pytest), not tmp_path.
        async def _patched_flush():
            safe_key = str(agent.session_key).replace(":", "_")
            sessions_dir = tmp_path / "sessions" / safe_key
            sessions_dir.mkdir(parents=True, exist_ok=True)
            path = sessions_dir / f"{agent._session_version}.json"
            dialogue_raw = [
                {
                    "id":           e.id,
                    "role":         e.role,
                    "content":      e.content,
                    "tool_calls":   e.tool_calls,
                    "tool_call_id": e.tool_call_id,
                    "index":        e.index,
                }
                for e in agent.context.dialogue
            ]
            data = {
                "session_key": str(agent.session_key),
                "version":     agent._session_version,
                "turn":        agent._turn_count,
                "saved_at":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "dialogue":    dialogue_raw,
            }
            path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        agent._flush_history = _patched_flush
        return agent

    return _factory


# ---------------------------------------------------------------------------
# Basic text reply
# ---------------------------------------------------------------------------

class TestBasicReply:
    @pytest.mark.asyncio
    async def test_simple_text_reply(self, make_agent):
        agent = make_agent(_text_stream("hello back"))
        chunks = await _collect(agent, _make_msg("hello"))
        assert _full_text(chunks) == "hello back"

    @pytest.mark.asyncio
    async def test_reply_is_agent_event(self, make_agent):
        agent = make_agent(_text_stream("hi"))
        chunks = await _collect(agent, _make_msg())
        assert all(isinstance(c, (AgentTextChunk, AgentTextFinal, AgentError)) for c in chunks)

    @pytest.mark.asyncio
    async def test_reply_session_key_matches(self, make_agent):
        sk = SessionKey.dm("u42")
        agent = make_agent(_text_stream("hi"))
        agent.session_key = sk
        chunks = await _collect(agent, _make_msg(session_key=sk))
        assert chunks[-1].session_key == sk

    @pytest.mark.asyncio
    async def test_multi_chunk_text_assembled(self, make_agent):
        agent = make_agent(_text_stream("part1", " ", "part2"))
        chunks = await _collect(agent, _make_msg())
        assert _full_text(chunks) == "part1 part2"

    @pytest.mark.asyncio
    async def test_turn_count_increments(self, make_agent):
        agent = make_agent(_text_stream("ok"))
        assert agent._turn_count == 0
        await _collect(agent, _make_msg("first"))
        assert agent._turn_count == 1
        await _collect(agent, _make_msg("second"))
        assert agent._turn_count == 2


# ---------------------------------------------------------------------------
# Context accumulation
# ---------------------------------------------------------------------------

class TestContextAccumulation:
    @pytest.mark.asyncio
    async def test_user_message_added_to_context(self, make_agent):
        agent = make_agent(_text_stream("ok"))
        await _collect(agent, _make_msg("test input"))
        roles = [e.role for e in agent.context.dialogue]
        assert "user" in roles

    @pytest.mark.asyncio
    async def test_assistant_reply_added_to_context(self, make_agent):
        agent = make_agent(_text_stream("my reply"))
        await _collect(agent, _make_msg("hi"))
        assistant_entries = [e for e in agent.context.dialogue if e.role == "assistant"]
        assert len(assistant_entries) == 1
        assert assistant_entries[0].content == "my reply"

    @pytest.mark.asyncio
    async def test_context_grows_across_turns(self, make_agent):
        agent = make_agent(_text_stream("ok"))
        await _collect(agent, _make_msg("turn one"))
        await _collect(agent, _make_msg("turn two"))
        assert len(agent.context.dialogue) == 4  # 2 user + 2 assistant


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

class TestToolExecution:
    @pytest.mark.asyncio
    async def test_tool_call_dispatched(self, make_agent):
        call_count = {"n": 0}

        async def stream(messages, tools=None):
            if call_count["n"] == 0:
                call_count["n"] += 1
                yield ToolCallAssembled(call_id="c1", tool_name="my_tool", args={"x": 1})
            else:
                yield TextDelta(text="done")

        agent = make_agent(stream)

        tool_called_with = []

        async def my_tool(x: int) -> str:
            """A test tool. Args: x: input."""
            tool_called_with.append(x)
            return "tool result"

        agent.tool_handler.register_tool(my_tool)
        await _collect(agent, _make_msg("use the tool"))
        assert tool_called_with == [1]

    @pytest.mark.asyncio
    async def test_tool_result_added_to_context(self, make_agent):
        call_count = {"n": 0}

        async def stream(messages, tools=None):
            if call_count["n"] == 0:
                call_count["n"] += 1
                yield ToolCallAssembled(call_id="c1", tool_name="my_tool", args={})
            else:
                yield TextDelta(text="done")

        agent = make_agent(stream)

        def my_tool() -> str:
            """Tool."""
            return "the answer"

        agent.tool_handler.register_tool(my_tool)
        await _collect(agent, _make_msg("go"))

        tool_results = [e for e in agent.context.dialogue if e.role == "tool"]
        assert len(tool_results) == 1
        assert "the answer" in tool_results[0].content

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error_result(self, make_agent):
        call_count = {"n": 0}

        async def stream(messages, tools=None):
            if call_count["n"] == 0:
                call_count["n"] += 1
                yield ToolCallAssembled(call_id="c1", tool_name="nonexistent", args={})
            else:
                yield TextDelta(text="done")

        agent = make_agent(stream)
        await _collect(agent, _make_msg("go"))

        tool_results = [e for e in agent.context.dialogue if e.role == "tool"]
        assert len(tool_results) == 1
        assert tool_results[0].content  # has some error message

    @pytest.mark.asyncio
    async def test_max_tool_cycles_respected(self, make_agent):
        """Agent must stop after max_tool_cycles even if LLM keeps calling tools."""
        always_tool_count = {"n": 0}

        async def always_tool(messages, tools=None):
            always_tool_count["n"] += 1
            yield ToolCallAssembled(
                call_id=f"c{always_tool_count['n']}",
                tool_name="loop_tool",
                args={},
            )

        agent = make_agent(always_tool)
        agent.config.max_tool_cycles = 3

        def loop_tool() -> str:
            """Loops forever."""
            return "keep going"

        agent.tool_handler.register_tool(loop_tool)
        chunks = await _collect(agent, _make_msg("go"))
        text_chunks = [c for c in chunks if isinstance(c, (AgentTextChunk, AgentTextFinal, AgentError))]
        assert any("cycle limit" in c.text.lower() or "tool" in c.text.lower() for c in text_chunks)
        assert always_tool_count["n"] <= 3


# ---------------------------------------------------------------------------
# LLM error handling and fallback
# ---------------------------------------------------------------------------

class TestLLMErrors:
    @pytest.mark.asyncio
    async def test_llm_error_surfaces_in_reply(self, make_agent):
        agent = make_agent(_error_stream("HTTP 500: server error"))
        chunks = await _collect(agent, _make_msg("hi"))
        assert "LLM error" in _full_text(chunks)

    @pytest.mark.asyncio
    async def test_fallback_used_on_error(self, make_agent):
        agent = make_agent(
            stream_fn=_error_stream("HTTP 429: rate limited"),
            fallback_stream_fn=_text_stream("fallback answer"),
            fallback_names=["fast"],
        )
        agent.config.llm.fallback_on.any_error = False
        agent.config.llm.fallback_on.http_codes = [429]

        chunks = await _collect(agent, _make_msg("hi"))
        assert _full_text(chunks) == "fallback answer"

    @pytest.mark.asyncio
    async def test_no_fallback_on_non_matching_code(self, make_agent):
        """A 404 error should NOT trigger fallback when only 429/500 are configured."""
        agent = make_agent(
            stream_fn=_error_stream("HTTP 404: not found"),
            fallback_stream_fn=_text_stream("should not see this"),
            fallback_names=["fast"],
        )
        agent.config.llm.fallback_on.any_error = False
        agent.config.llm.fallback_on.http_codes = [429, 500]

        chunks = await _collect(agent, _make_msg("hi"))
        assert "should not see this" not in _full_text(chunks)
        assert "LLM error" in _full_text(chunks)

    @pytest.mark.asyncio
    async def test_any_error_fallback(self, make_agent):
        """any_error=True should fall back on any LLMError regardless of code."""
        agent = make_agent(
            stream_fn=_error_stream("Connection failed: timeout"),
            fallback_stream_fn=_text_stream("fallback worked"),
            fallback_names=["fast"],
        )
        agent.config.llm.fallback_on.any_error = True

        chunks = await _collect(agent, _make_msg("hi"))
        assert _full_text(chunks) == "fallback worked"


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    @pytest.mark.asyncio
    async def test_reset_clears_context(self, make_agent):
        agent = make_agent(_text_stream("ok"))
        await _collect(agent, _make_msg("hi"))
        assert len(agent.context.dialogue) > 0

        agent.reset()
        assert agent.context.dialogue == []

    @pytest.mark.asyncio
    async def test_reset_zeroes_turn_count(self, make_agent):
        agent = make_agent(_text_stream("ok"))
        await _collect(agent, _make_msg("hi"))
        assert agent._turn_count == 1

        agent.reset()
        assert agent._turn_count == 0

    @pytest.mark.asyncio
    async def test_reset_increments_version(self, make_agent):
        agent = make_agent(_text_stream("ok"))
        v_before = agent._session_version
        agent.reset()
        assert agent._session_version == v_before + 1


# ---------------------------------------------------------------------------
# History persistence
# ---------------------------------------------------------------------------

class TestHistoryPersistence:
    @pytest.mark.asyncio
    async def test_flush_writes_session_file(self, make_agent, tmp_path):
        agent = make_agent(_text_stream("ok"))
        await _collect(agent, _make_msg("hi"))

        safe_key = str(agent.session_key).replace(":", "_")
        session_dir = tmp_path / "sessions" / safe_key
        files = list(session_dir.glob("*.json")) if session_dir.exists() else []
        assert len(files) >= 1

    @pytest.mark.asyncio
    async def test_flushed_file_contains_dialogue(self, make_agent, tmp_path):
        agent = make_agent(_text_stream("my reply"))
        await _collect(agent, _make_msg("my question"))

        safe_key = str(agent.session_key).replace(":", "_")
        session_dir = tmp_path / "sessions" / safe_key
        files = list(session_dir.glob("*.json"))
        assert files

        data = json.loads(files[0].read_text())
        contents = [e["content"] for e in data["dialogue"]]
        assert any("my question" in c for c in contents)
        assert any("my reply" in c for c in contents)

    @pytest.mark.asyncio
    async def test_version_increments_in_file_after_reset(self, make_agent, tmp_path):
        agent = make_agent(_text_stream("ok"))
        await _collect(agent, _make_msg("hi"))
        v1 = agent._session_version

        agent.reset()
        await _collect(agent, _make_msg("hi again"))
        v2 = agent._session_version

        assert v2 == v1 + 1