"""
tests/test_gateway.py

Tests for Gateway, SessionRouter, and Lane.

We avoid starting real AgentLoops (which need config, LLMs, modules, etc.)
by monkey-patching Lane.__post_init__ and AgentLoop.run so the gateway's
routing and dispatch logic can be tested in pure isolation.

Run with:
    python -m pytest tests/
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from contracts import (
    SessionKey, UserIdentity, InboundMessage, OutboundReply,
    Platform, ContentType, ChatType,
)
from gateway import Gateway, Lane, SessionRouter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config():
    """Minimal Config stub — gateway only needs a few fields."""
    cfg = MagicMock()
    cfg.models = {"primary": MagicMock()}
    cfg.llm.primary = "primary"
    cfg.llm.fallback = []
    cfg.context = 4096
    cfg.max_tool_cycles = 5
    cfg.memory.workspace_path = "/tmp/tinyctx_test"
    return cfg


def _make_user(platform=Platform.CLI, user_id="u1", username="alice"):
    return UserIdentity(platform=platform, user_id=user_id, username=username)


def _make_msg(session_key, text="hello", user_id="u1", platform=Platform.CLI):
    return InboundMessage(
        session_key=session_key,
        author=_make_user(platform=platform, user_id=user_id),
        content_type=ContentType.TEXT,
        text=text,
        message_id="msg-1",
        timestamp=0.0,
    )


def _make_reply(session_key, text="hi"):
    return OutboundReply(
        session_key=session_key,
        text=text,
        reply_to_message_id="msg-1",
        trace_id="trace-1",
        is_partial=False,
    )


# ---------------------------------------------------------------------------
# Fixture: a Gateway with AgentLoop stubbed out
# ---------------------------------------------------------------------------

@pytest.fixture
def gw():
    """
    Gateway with AgentLoop creation patched to a no-op stub.
    The stub's run() is an async generator that yields nothing by default —
    tests can override it per-scenario.
    """
    cfg = _make_config()

    async def _noop_run(msg):
        return
        yield  # make it an async generator

    with patch("gateway.AgentLoop") as MockLoop:
        instance = MagicMock()
        instance.run = _noop_run
        instance.gateway = None
        instance.reset = MagicMock()
        MockLoop.return_value = instance
        gw = Gateway(config=cfg)
        gw._mock_loop_instance = instance
        yield gw


# ---------------------------------------------------------------------------
# Reply handler registration
# ---------------------------------------------------------------------------

class TestReplyHandlerRegistration:
    def test_register_handler(self, gw):
        handler = AsyncMock()
        gw.register_reply_handler("cli", handler)
        assert "cli" in gw._reply_handlers

    def test_register_multiple_platforms(self, gw):
        gw.register_reply_handler("cli", AsyncMock())
        gw.register_reply_handler("discord", AsyncMock())
        assert "cli" in gw._reply_handlers
        assert "discord" in gw._reply_handlers


# ---------------------------------------------------------------------------
# Session routing — lane creation
# ---------------------------------------------------------------------------

class TestSessionRouting:
    @pytest.mark.asyncio
    async def test_dm_creates_lane(self, gw):
        sk = SessionKey.dm("u1")
        gw.register_reply_handler("cli", AsyncMock())
        await gw.push(_make_msg(sk))
        await asyncio.sleep(0)  # let the event loop tick
        assert sk in gw._router._lanes

    @pytest.mark.asyncio
    async def test_group_creates_lane(self, gw):
        sk = SessionKey.group(Platform.DISCORD, "ch1")
        gw.register_reply_handler("discord", AsyncMock())
        await gw.push(_make_msg(sk, platform=Platform.DISCORD))
        await asyncio.sleep(0)
        assert sk in gw._router._lanes

    @pytest.mark.asyncio
    async def test_same_session_reuses_lane(self, gw):
        sk = SessionKey.dm("u1")
        gw.register_reply_handler("cli", AsyncMock())
        await gw.push(_make_msg(sk, text="first"))
        await gw.push(_make_msg(sk, text="second"))
        await asyncio.sleep(0)
        assert len(gw._router._lanes) == 1

    @pytest.mark.asyncio
    async def test_different_dm_sessions_get_different_lanes(self, gw):
        sk1 = SessionKey.dm("u1")
        sk2 = SessionKey.dm("u2")
        gw.register_reply_handler("cli", AsyncMock())
        await gw.push(_make_msg(sk1, user_id="u1"))
        await gw.push(_make_msg(sk2, user_id="u2"))
        await asyncio.sleep(0)
        assert len(gw._router._lanes) == 2

    @pytest.mark.asyncio
    async def test_active_sessions_reflects_lanes(self, gw):
        sk1 = SessionKey.dm("u1")
        sk2 = SessionKey.dm("u2")
        gw.register_reply_handler("cli", AsyncMock())
        await gw.push(_make_msg(sk1, user_id="u1"))
        await gw.push(_make_msg(sk2, user_id="u2"))
        await asyncio.sleep(0)
        assert set(gw.active_sessions) == {sk1, sk2}


# ---------------------------------------------------------------------------
# DM platform tracking
# ---------------------------------------------------------------------------

class TestDMPlatformTracking:
    @pytest.mark.asyncio
    async def test_dm_platform_recorded_on_push(self, gw):
        sk = SessionKey.dm("u1")
        gw.register_reply_handler("cli", AsyncMock())
        await gw.push(_make_msg(sk, platform=Platform.CLI))
        assert gw._dm_platforms.get(sk) == "cli"

    @pytest.mark.asyncio
    async def test_dm_platform_updated_on_second_push(self, gw):
        sk = SessionKey.dm("u1")
        gw.register_reply_handler("cli", AsyncMock())
        gw.register_reply_handler("discord", AsyncMock())
        await gw.push(_make_msg(sk, platform=Platform.CLI))
        # Same user now messages from Discord
        await gw.push(InboundMessage(
            session_key=sk,
            author=UserIdentity(platform=Platform.DISCORD, user_id="u1", username="alice"),
            content_type=ContentType.TEXT,
            text="hello from discord",
            message_id="msg-2",
            timestamp=1.0,
        ))
        assert gw._dm_platforms.get(sk) == "discord"


# ---------------------------------------------------------------------------
# Reply dispatch
# ---------------------------------------------------------------------------

class TestReplyDispatch:
    @pytest.mark.asyncio
    async def test_reply_dispatched_to_correct_handler(self, gw):
        sk = SessionKey.dm("u1")
        received = []

        async def handler(reply):
            received.append(reply)

        gw.register_reply_handler("cli", handler)
        gw._dm_platforms[sk] = "cli"

        reply = _make_reply(sk)
        await gw._dispatch_reply(reply)
        assert received == [reply]

    @pytest.mark.asyncio
    async def test_reply_dropped_if_no_platform_known(self, gw, caplog):
        sk = SessionKey.dm("u1")
        # No platform registered for this session
        reply = _make_reply(sk)
        # Should not raise — just logs an error
        await gw._dispatch_reply(reply)
        assert "Cannot determine platform" in caplog.text

    @pytest.mark.asyncio
    async def test_reply_dropped_if_no_handler_registered(self, gw, caplog):
        sk = SessionKey.dm("u1")
        gw._dm_platforms[sk] = "matrix"
        # No handler registered for "matrix"
        reply = _make_reply(sk)
        await gw._dispatch_reply(reply)
        assert "No reply handler" in caplog.text

    @pytest.mark.asyncio
    async def test_group_reply_uses_platform_from_session_key(self, gw):
        sk = SessionKey.group(Platform.DISCORD, "ch1")
        received = []

        async def handler(reply):
            received.append(reply)

        gw.register_reply_handler("discord", handler)
        reply = _make_reply(sk)
        await gw._dispatch_reply(reply)
        assert received == [reply]

    @pytest.mark.asyncio
    async def test_handler_exception_does_not_propagate(self, gw):
        sk = SessionKey.dm("u1")
        gw._dm_platforms[sk] = "cli"

        async def bad_handler(reply):
            raise RuntimeError("handler blew up")

        gw.register_reply_handler("cli", bad_handler)
        # Should not raise out of _dispatch_reply
        await gw._dispatch_reply(_make_reply(sk))


# ---------------------------------------------------------------------------
# reset_session
# ---------------------------------------------------------------------------

class TestResetSession:
    @pytest.mark.asyncio
    async def test_reset_existing_session_calls_loop_reset(self, gw):
        sk = SessionKey.dm("u1")
        gw.register_reply_handler("cli", AsyncMock())
        await gw.push(_make_msg(sk))
        await asyncio.sleep(0)

        gw.reset_session(sk)
        gw._mock_loop_instance.reset.assert_called_once()

    def test_reset_nonexistent_session_is_noop(self, gw):
        sk = SessionKey.dm("nobody")
        # Should not raise
        gw.reset_session(sk)


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

class TestShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_closes_all_lanes(self, gw):
        sk1 = SessionKey.dm("u1")
        sk2 = SessionKey.dm("u2")
        gw.register_reply_handler("cli", AsyncMock())
        await gw.push(_make_msg(sk1, user_id="u1"))
        await gw.push(_make_msg(sk2, user_id="u2"))
        await asyncio.sleep(0)

        await gw.shutdown()
        assert gw._router._lanes == {}

    @pytest.mark.asyncio
    async def test_active_sessions_empty_after_shutdown(self, gw):
        sk = SessionKey.dm("u1")
        gw.register_reply_handler("cli", AsyncMock())
        await gw.push(_make_msg(sk))
        await asyncio.sleep(0)

        await gw.shutdown()
        assert gw.active_sessions == []