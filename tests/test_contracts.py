"""
tests/test_contracts.py

Tests for pure data types in contracts.py — SessionKey, UserIdentity,
InboundMessage, OutboundReply, ToolCall, ToolResult.

Run with:
    pytest tests/
"""
import pytest
from contracts import (
    SessionKey, UserIdentity, InboundMessage, OutboundReply,
    ToolCall, ToolResult,
    Platform, ContentType, ChatType,
)


# ---------------------------------------------------------------------------
# SessionKey
# ---------------------------------------------------------------------------

class TestSessionKey:
    def test_dm_session_key(self):
        key = SessionKey.dm("user123")
        assert key.chat_type == ChatType.DM
        assert key.conversation_id == "user123"
        assert key.platform is None

    def test_group_session_key(self):
        key = SessionKey.group(Platform.DISCORD, "channel456")
        assert key.chat_type == ChatType.GROUP
        assert key.conversation_id == "channel456"
        assert key.platform == Platform.DISCORD

    def test_group_without_platform_raises(self):
        with pytest.raises(ValueError, match="platform"):
            SessionKey(chat_type=ChatType.GROUP, conversation_id="ch1", platform=None)

    def test_dm_str(self):
        key = SessionKey.dm("user123")
        assert str(key) == "dm:user123"

    def test_group_str(self):
        key = SessionKey.group(Platform.DISCORD, "ch1")
        assert str(key) == "group:discord:ch1"

    def test_dm_keys_are_hashable(self):
        key = SessionKey.dm("u1")
        d = {key: "value"}
        assert d[key] == "value"

    def test_dm_keys_are_equal_with_same_id(self):
        a = SessionKey.dm("u1")
        b = SessionKey.dm("u1")
        assert a == b

    def test_group_keys_differ_by_platform(self):
        discord = SessionKey.group(Platform.DISCORD, "ch1")
        matrix  = SessionKey.group(Platform.MATRIX,  "ch1")
        assert discord != matrix

    def test_frozen_immutability(self):
        key = SessionKey.dm("u1")
        with pytest.raises((AttributeError, TypeError)):
            key.conversation_id = "changed"  # type: ignore


# ---------------------------------------------------------------------------
# ToolCall
# ---------------------------------------------------------------------------

class TestToolCall:
    def test_make_generates_unique_ids(self):
        a = ToolCall.make("fn", {})
        b = ToolCall.make("fn", {})
        assert a.call_id != b.call_id

    def test_make_stores_name_and_args(self):
        tc = ToolCall.make("search", {"q": "cats"})
        assert tc.tool_name == "search"
        assert tc.args == {"q": "cats"}

    def test_frozen(self):
        tc = ToolCall(call_id="x", tool_name="fn", args={})
        with pytest.raises((AttributeError, TypeError)):
            tc.tool_name = "other"  # type: ignore


# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------

class TestToolResult:
    def test_defaults(self):
        r = ToolResult(call_id="x", tool_name="fn", output="ok")
        assert r.is_error is False

    def test_error_flag(self):
        r = ToolResult(call_id="x", tool_name="fn", output="boom", is_error=True)
        assert r.is_error is True


# ---------------------------------------------------------------------------
# Platform enum
# ---------------------------------------------------------------------------

class TestPlatform:
    def test_platform_values(self):
        assert Platform.CLI.value == "cli"
        assert Platform.DISCORD.value == "discord"
        assert Platform.MATRIX.value == "matrix"

    def test_platform_from_value(self):
        assert Platform("cli") == Platform.CLI