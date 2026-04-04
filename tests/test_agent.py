"""
tests/test_agent.py

Tests for AgentLoop — the 6-stage execution loop.

The LLM is fully stubbed via async generator functions so no network calls
are made. Modules are not loaded (MODULES_DIR is patched to a non-existent
path). Each make_agent() call gets a completely fresh AgentLoop with its own
unique session key so tests never bleed into each other.

Phase 1 tree-refactor notes:
  - _flush_history, _restore_history, _load_latest_version, next_session
    are gone. Session-file persistence tests have been removed.
  - reset() only clears in-memory state; it no longer writes to disk.
  - Tests that validated those behaviours now live in test_tree_refactor.py.

Run with:
    python -m pytest tests/
"""
import asyncio
import time
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from contracts import (
    UserIdentity, InboundMessage,
    AgentTextChunk, AgentTextFinal, AgentError,
    Platform, ContentType, ToolCall, ToolResult,
)
from context import HistoryEntry
from ai import TextDelta, ToolCallAssembled, LLMError
from db import ConversationDB
from compact import COMPACT_SYSTEM_PROMPT, COMPACT_USER_PROMPT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path):
    cfg = MagicMock()
    primary_mc = MagicMock()
    primary_mc.is_embedding = False
    fast_mc = MagicMock()
    fast_mc.is_embedding = False
    cfg.models = {"primary": primary_mc, "fast": fast_mc}
    cfg.llm.primary = "primary"
    cfg.llm.fallback = []
    cfg.llm.fallback_on.any_error = False
    cfg.llm.fallback_on.http_codes = [429, 500]
    cfg.context = 4096
    cfg.max_tool_cycles = 5
    cfg.workspace.path = str(tmp_path)
    cfg.attachments = MagicMock()
    cfg.get_model_config = MagicMock(return_value=MagicMock(vision=False))
    return cfg


def _make_cursor(tmp_path) -> str:
    """Create a fresh DB cursor node in tmp_path and return its node_id."""
    db   = ConversationDB(tmp_path / "agent.db")
    root = db.get_root()
    node = db.add_node(parent_id=root.id, role="system", content="session:test")
    return node.id


def _make_msg(text="hello", node_id=None, tmp_path=None):
    nid = node_id or "test-node-id"
    return InboundMessage(
        tail_node_id=nid,
        author=UserIdentity(platform=Platform.CLI, user_id="u1", username="alice"),
        content_type=ContentType.TEXT,
        text=text,
        message_id="msg-1",
        timestamp=0.0,
    )


async def _collect(agent, msg):
    """Run the agent and collect all reply events."""
    chunks = []
    async for chunk in agent.run(msg):
        chunks.append(chunk)
    return chunks


def _full_text(chunks):
    return "".join(c.text if hasattr(c, "text") else c.message for c in chunks)


def _text_stream(*texts):
    async def _gen(messages, tools=None):
        for t in texts:
            yield TextDelta(text=t)
    return _gen


def _error_stream(message):
    async def _gen(messages, tools=None):
        yield LLMError(message=message)
    return _gen


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def make_agent(tmp_path):
    """
    Factory fixture — call make_agent(stream_fn) to get a fresh AgentLoop.

    Each call creates a unique cursor node in the shared agent.db so DB state
    never bleeds between tests. The workspace is tmp_path.
    """
    def _factory(stream_fn=None, fallback_stream_fn=None, fallback_names=None):
        from agent import AgentLoop

        node_id = _make_cursor(tmp_path)
        cfg = _make_config(tmp_path)

        if fallback_names:
            cfg.llm.fallback = fallback_names
            cfg.llm.fallback_on.any_error = True

        primary_llm = MagicMock()
        primary_llm.stream = stream_fn or _text_stream("default reply")

        fallback_llm = MagicMock()
        fallback_llm.stream = fallback_stream_fn or _text_stream("fallback reply")

        with patch("agent.MODULES_DIR", Path("/nonexistent")):
            with patch("agent._build_llm", return_value=primary_llm):
                agent = AgentLoop(tail_node_id=node_id, config=cfg)

        agent._models["primary"] = primary_llm
        if fallback_names:
            for name in fallback_names:
                agent._models[name] = fallback_llm

        return agent

    return _factory


# ---------------------------------------------------------------------------
# Basic text reply
# ---------------------------------------------------------------------------

class TestBasicReply:
    @pytest.mark.asyncio
    async def test_simple_text_reply(self, make_agent):
        agent = make_agent(_text_stream("hello back"))
        chunks = await _collect(agent, _make_msg("hello", node_id=agent.tail_node_id))
        assert _full_text(chunks) == "hello back"

    @pytest.mark.asyncio
    async def test_reply_is_agent_event(self, make_agent):
        agent = make_agent(_text_stream("hi"))
        chunks = await _collect(agent, _make_msg(node_id=agent.tail_node_id))
        assert all(isinstance(c, (AgentTextChunk, AgentTextFinal, AgentError)) for c in chunks)

    @pytest.mark.asyncio
    async def test_reply_tail_node_id_matches(self, make_agent):
        agent = make_agent(_text_stream("hi"))
        node_id = agent.tail_node_id
        chunks = await _collect(agent, _make_msg(node_id=node_id))
        assert chunks[-1].tail_node_id is not None

    @pytest.mark.asyncio
    async def test_multi_chunk_text_assembled(self, make_agent):
        agent = make_agent(_text_stream("part1", " ", "part2"))
        chunks = await _collect(agent, _make_msg(node_id=agent.tail_node_id))
        assert _full_text(chunks) == "part1 part2"

    @pytest.mark.asyncio
    async def test_turn_count_increments(self, make_agent):
        agent = make_agent(_text_stream("ok"))
        assert agent._turn_count == 0
        await _collect(agent, _make_msg("first", node_id=agent.tail_node_id))
        assert agent._turn_count == 1
        await _collect(agent, _make_msg("second", node_id=agent.tail_node_id))
        assert agent._turn_count == 2


# ---------------------------------------------------------------------------
# Context accumulation
# ---------------------------------------------------------------------------

class TestContextAccumulation:
    @pytest.mark.asyncio
    async def test_user_message_added_to_context(self, make_agent):
        agent = make_agent(_text_stream("ok"))
        await _collect(agent, _make_msg("test input", node_id=agent.tail_node_id))
        roles = [e.role for e in agent.context.dialogue]
        assert "user" in roles

    @pytest.mark.asyncio
    async def test_assistant_reply_added_to_context(self, make_agent):
        agent = make_agent(_text_stream("my reply"))
        await _collect(agent, _make_msg("hi", node_id=agent.tail_node_id))
        assistant_entries = [e for e in agent.context.dialogue if e.role == "assistant"]
        assert len(assistant_entries) == 1
        assert assistant_entries[0].content == "my reply"

    @pytest.mark.asyncio
    async def test_context_grows_across_turns(self, make_agent):
        agent = make_agent(_text_stream("ok"))
        await _collect(agent, _make_msg("turn one", node_id=agent.tail_node_id))
        await _collect(agent, _make_msg("turn two", node_id=agent.tail_node_id))
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
        await _collect(agent, _make_msg("use the tool", node_id=agent.tail_node_id))
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
        await _collect(agent, _make_msg("go", node_id=agent.tail_node_id))

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
        await _collect(agent, _make_msg("go", node_id=agent.tail_node_id))

        tool_results = [e for e in agent.context.dialogue if e.role == "tool"]
        assert len(tool_results) == 1
        assert tool_results[0].content  # some error message

    @pytest.mark.asyncio
    async def test_max_tool_cycles_respected(self, make_agent):
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
        chunks = await _collect(agent, _make_msg("go", node_id=agent.tail_node_id))
        text_chunks = [c for c in chunks if isinstance(c, (AgentTextChunk, AgentTextFinal, AgentError))]
        assert any("cycle limit" in c.text.lower() or "tool" in c.text.lower() for c in text_chunks)
        assert always_tool_count["n"] <= 3

    @pytest.mark.asyncio
    async def test_identical_tool_call_reuses_cached_result_within_turn(self, make_agent):
        call_count = {"n": 0}

        async def stream(messages, tools=None):
            if call_count["n"] == 0:
                call_count["n"] += 1
                yield ToolCallAssembled(call_id="c1", tool_name="my_tool", args={"url": "https://example.com"})
            elif call_count["n"] == 1:
                call_count["n"] += 1
                yield ToolCallAssembled(call_id="c2", tool_name="my_tool", args={"url": "https://example.com"})
            else:
                yield TextDelta(text="done")

        agent = make_agent(stream)

        tool_invocations = {"n": 0}

        def my_tool(url: str) -> str:
            """Fetch something."""
            tool_invocations["n"] += 1
            return f"result for {url}"

        agent.tool_handler.register_tool(my_tool)
        await _collect(agent, _make_msg("go", node_id=agent.tail_node_id))

        tool_results = [e for e in agent.context.dialogue if e.role == "tool"]
        assert tool_invocations["n"] == 1
        assert len(tool_results) == 2
        assert "result for https://example.com" in tool_results[0].content
        assert "cached identical tool call reused earlier result" in tool_results[1].content

    @pytest.mark.asyncio
    async def test_different_tool_args_do_not_reuse_cached_result(self, make_agent):
        call_count = {"n": 0}

        async def stream(messages, tools=None):
            if call_count["n"] == 0:
                call_count["n"] += 1
                yield ToolCallAssembled(call_id="c1", tool_name="my_tool", args={"x": 1})
            elif call_count["n"] == 1:
                call_count["n"] += 1
                yield ToolCallAssembled(call_id="c2", tool_name="my_tool", args={"x": 2})
            else:
                yield TextDelta(text="done")

        agent = make_agent(stream)

        seen = []

        def my_tool(x: int) -> str:
            """Tool."""
            seen.append(x)
            return str(x)

        agent.tool_handler.register_tool(my_tool)
        await _collect(agent, _make_msg("go", node_id=agent.tail_node_id))

        assert seen == [1, 2]


# ---------------------------------------------------------------------------
# LLM error handling and fallback
# ---------------------------------------------------------------------------

class TestLLMErrors:
    @pytest.mark.asyncio
    async def test_llm_error_surfaces_in_reply(self, make_agent):
        agent = make_agent(_error_stream("HTTP 500: server error"))
        chunks = await _collect(agent, _make_msg("hi", node_id=agent.tail_node_id))
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

        chunks = await _collect(agent, _make_msg("hi", node_id=agent.tail_node_id))
        assert _full_text(chunks) == "fallback answer"

    @pytest.mark.asyncio
    async def test_no_fallback_on_non_matching_code(self, make_agent):
        agent = make_agent(
            stream_fn=_error_stream("HTTP 404: not found"),
            fallback_stream_fn=_text_stream("should not see this"),
            fallback_names=["fast"],
        )
        agent.config.llm.fallback_on.any_error = False
        agent.config.llm.fallback_on.http_codes = [429, 500]

        chunks = await _collect(agent, _make_msg("hi", node_id=agent.tail_node_id))
        assert "should not see this" not in _full_text(chunks)
        assert "LLM error" in _full_text(chunks)

    @pytest.mark.asyncio
    async def test_any_error_fallback(self, make_agent):
        agent = make_agent(
            stream_fn=_error_stream("Connection failed: timeout"),
            fallback_stream_fn=_text_stream("fallback worked"),
            fallback_names=["fast"],
        )
        agent.config.llm.fallback_on.any_error = True

        chunks = await _collect(agent, _make_msg("hi", node_id=agent.tail_node_id))
        assert _full_text(chunks) == "fallback worked"


# ---------------------------------------------------------------------------
# Compaction
# ---------------------------------------------------------------------------

class TestCompaction:
    @pytest.mark.asyncio
    async def test_context_compacts_before_normal_inference(self, make_agent):
        calls = []

        async def stream(messages, tools=None):
            calls.append({"messages": messages, "tools": tools})
            if messages and messages[0].get("content") == COMPACT_SYSTEM_PROMPT:
                yield TextDelta(text="- durable older context")
                return
            yield TextDelta(text="final after compact")

        agent = make_agent(stream)
        agent.config.context = 120

        for i in range(6):
            agent.context.add(HistoryEntry.user(f"user-{i} " + ("x" * 80)))
            agent.context.add(HistoryEntry.assistant(f"assistant-{i} " + ("y" * 80)))

        chunks = await _collect(
            agent,
            _make_msg("latest request " + ("z" * 80), node_id=agent.tail_node_id),
        )

        assert _full_text(chunks) == "final after compact"

        compact_call = next(
            call for call in calls if call["messages"][0].get("content") == COMPACT_SYSTEM_PROMPT
        )
        assert compact_call["tools"] is None
        assert compact_call["messages"][-1]["content"] == COMPACT_USER_PROMPT

        active_users = [
            entry.content
            for entry in agent.context.dialogue
            if entry.role == "user" and isinstance(entry.content, str)
        ]
        assert any(content.startswith("Compact summary:\n- durable older context") for content in active_users)
        assert not any(content.startswith("user-0 ") for content in active_users)


# ---------------------------------------------------------------------------
# Background branches (Phase 3)
# ---------------------------------------------------------------------------

class TestBackgroundBranches:
    def test_queue_background_branch_accumulates(self, make_agent):
        """queue_background_branch() appends node_ids without side-effects."""
        agent = make_agent(_text_stream("ok"))
        assert agent._pending_background_branches == []
        agent.queue_background_branch("node-a")
        agent.queue_background_branch("node-b")
        assert agent._pending_background_branches == ["node-a", "node-b"]

    @pytest.mark.asyncio
    async def test_pending_branches_cleared_after_turn(self, make_agent, tmp_path):
        """After a turn completes, _pending_background_branches is drained."""
        agent = make_agent(_text_stream("hi"))
        # Pre-populate with a fake node_id — we don't want _run_background
        # to actually launch (it would fail on a bad node_id), so patch it.
        # Use AsyncMock so ensure_future receives an awaitable that won't leak.
        agent._run_background = AsyncMock()

        # Queue a branch before the turn fires
        agent._pending_background_branches.append("fake-branch-id")

        with patch("asyncio.ensure_future") as mock_ef:
            await _collect(agent, _make_msg("hi", node_id=agent.tail_node_id))

        # List must be empty after the turn regardless of whether tasks ran
        assert agent._pending_background_branches == []

    @pytest.mark.asyncio
    async def test_run_background_writes_nodes_to_db(self, make_agent, tmp_path):
        """
        _run_background() runs a real synthetic AgentLoop turn.
        The branch node must gain at least an assistant response in the DB.
        """
        from db import ConversationDB

        bg_llm = MagicMock()
        bg_llm.stream = _text_stream("background done")

        # Both the outer agent and the inner background AgentLoop call _build_llm;
        # keep the patch active for both construction and the _run_background call.
        with patch("agent.MODULES_DIR", Path("/nonexistent")):
            with patch("agent._build_llm", return_value=bg_llm):
                agent = make_agent(_text_stream("main reply"))

                # Create a branch opening node off the current tail
                db = ConversationDB(tmp_path / "agent.db")
                branch_node = db.add_node(
                    parent_id=agent._tail_node_id,
                    role="user",
                    content="consolidate memory",
                )

                await agent._run_background(branch_node.id)

        children = db.get_children(branch_node.id)
        assert children, "No children written under branch_node — background loop produced no output"
        bg_ancestors = db.get_ancestors(children[0].id)
        assistant_nodes = [n for n in bg_ancestors if n.role == "assistant"]
        assert len(assistant_nodes) >= 1
        assert "background done" in assistant_nodes[-1].content

    @pytest.mark.asyncio
    async def test_run_background_does_not_move_main_cursor(self, make_agent, tmp_path):
        """Running a background branch must leave the main agent cursor untouched."""
        from db import ConversationDB

        agent = make_agent(_text_stream("main reply"))
        tail_before = agent._tail_node_id

        db = ConversationDB(tmp_path / "agent.db")
        branch_node = db.add_node(
            parent_id=tail_before,
            role="user",
            content="do background work",
        )

        bg_llm = MagicMock()
        bg_llm.stream = _text_stream("done")

        with patch("agent.MODULES_DIR", Path("/nonexistent")):
            with patch("agent._build_llm", return_value=bg_llm):
                await agent._run_background(branch_node.id)

        assert agent._tail_node_id == tail_before


# ---------------------------------------------------------------------------
# Reset (in-memory only after tree refactor)
# ---------------------------------------------------------------------------

class TestReset:
    @pytest.mark.asyncio
    async def test_reset_clears_context(self, make_agent):
        agent = make_agent(_text_stream("ok"))
        await _collect(agent, _make_msg("hi", node_id=agent.tail_node_id))
        assert len(agent.context.dialogue) > 0

        agent.reset()
        assert agent.context.dialogue == []

    @pytest.mark.asyncio
    async def test_reset_zeroes_turn_count(self, make_agent):
        agent = make_agent(_text_stream("ok"))
        await _collect(agent, _make_msg("hi", node_id=agent.tail_node_id))
        assert agent._turn_count == 1

        agent.reset()
        assert agent._turn_count == 0

    @pytest.mark.asyncio
    async def test_reset_does_not_delete_db_nodes(self, make_agent, tmp_path):
        """reset() is in-memory only — the tree in agent.db must survive."""
        agent = make_agent(_text_stream("ok"))
        await _collect(agent, _make_msg("hi", node_id=agent.tail_node_id))
        tail_before = agent._tail_node_id

        agent.reset()

        db = ConversationDB(tmp_path / "agent.db")
        assert db.get_node(tail_before) is not None
