"""
router.py — Async session router and lane queues.
No LLM calls, no tools, no memory. Pure routing and async primitives.
Imports only from contracts.py, agent.py, config.py, and stdlib.

Group chat routing
------------------
Group sessions get a GroupLane wrapper around the inner Lane.
GroupLane owns all group-specific logic that used to live in bridges:

  - Trigger detection   (mention / prefix / always)
  - Text stripping      (remove @mention and prefix from the forwarded text)
  - Non-trigger buffer  (collect context lines until a trigger arrives)
  - Timeout flush       (optional: flush buffer after N seconds without trigger)

Bridges pass *raw* (un-stripped) text and attach a GroupPolicy describing
how the bot should activate in that room. GroupLane applies the policy and
only enqueues a message to the inner Lane when appropriate.

Runtime activation toggle
-------------------------
Router.set_group_activation(key, mode) replaces the GroupPolicy activation
field at runtime. This lets bridges implement /activation commands without
restarting or reconfiguring.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Callable, Awaitable

from contracts import (
    InboundMessage, AgentEvent, SessionKey, ChatType,
    GroupPolicy, ActivationMode,
)
from agent import AgentLoop
from config import Config

logger = logging.getLogger(__name__)

EventHandler = Callable[[AgentEvent], Awaitable[None]]


# ---------------------------------------------------------------------------
# Lane
# ---------------------------------------------------------------------------

LANE_QUEUE_MAX = 32  # max pending turns per lane before backpressure kicks in


@dataclass
class Lane:
    session_key:      SessionKey
    config:           Config
    event_handler:    EventHandler
    router:           "Router"
    version_override: int | None = None
    loop:             AgentLoop = field(init=False)
    queue:            asyncio.Queue = field(init=False)
    _worker:          asyncio.Task | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.queue = asyncio.Queue(maxsize=LANE_QUEUE_MAX)
        self.loop = AgentLoop(
            session_key=self.session_key,
            config=self.config,
            version_override=self.version_override,
        )
        self.loop.gateway = self.router

    def start(self) -> None:
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(
                self._drain(), name=f"lane:{self.session_key}"
            )

    async def enqueue(self, msg: InboundMessage) -> bool:
        """Enqueue a message. Returns False and drops the message if the queue is full."""
        try:
            self.queue.put_nowait(msg)
            return True
        except asyncio.QueueFull:
            logger.warning(
                "Lane %s queue full (%d/%d) — dropping message %s",
                self.session_key, self.queue.qsize(), LANE_QUEUE_MAX, msg.trace_id,
            )
            return False

    async def _drain(self) -> None:
        while True:
            msg = await self.queue.get()
            try:
                async for event in self.loop.run(msg):
                    try:
                        await self.event_handler(event)
                    except Exception:
                        logger.exception("Event handler raised for event %s", event.trace_id)
            except Exception:
                logger.exception(
                    "AgentLoop raised for %s — lane continues", self.session_key
                )
            finally:
                self.queue.task_done()

    def reset(self) -> None:
        """Reset the AgentLoop for this lane — clears conversation context."""
        self.loop.reset()

    async def stop(self) -> None:
        if self._worker and not self._worker.done():
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass


# ---------------------------------------------------------------------------
# GroupLane helpers
# ---------------------------------------------------------------------------

def _gp_strip_trigger(text: str, policy: GroupPolicy) -> str:
    """Remove bot mention(s) and command prefix from text."""
    if policy.bot_mxid:
        text = text.replace(policy.bot_mxid, "")
    if policy.bot_localpart:
        text = text.replace(f"@{policy.bot_localpart}", "")
    if text.startswith(policy.trigger_prefix):
        text = text[len(policy.trigger_prefix):]
    return text.strip()


def _gp_replace_text(msg: InboundMessage, new_text: str) -> InboundMessage:
    """Return a copy of msg with text replaced (frozen dataclass rebuild)."""
    return InboundMessage(
        session_key=msg.session_key,
        author=msg.author,
        content_type=msg.content_type,
        text=new_text,
        message_id=msg.message_id,
        timestamp=msg.timestamp,
        reply_to_id=msg.reply_to_id,
        attachments=msg.attachments,
        trace_id=msg.trace_id,
        group_policy=msg.group_policy,
    )


# ---------------------------------------------------------------------------
# GroupLane
# ---------------------------------------------------------------------------

class GroupLane:
    """
    Wraps a Lane for group sessions and applies GroupPolicy:

      - ALWAYS:  every message is forwarded immediately (no trigger check).
      - MENTION: message must @mention the bot or start with trigger_prefix.
      - PREFIX:  message must start with trigger_prefix only.

    Non-trigger messages are buffered as context lines. When a trigger
    arrives the buffer is flushed and prepended to the trigger text as
    [DisplayName]: <text> lines so the agent sees full conversation context.

    The inner Lane and its AgentLoop are the same objects the rest of the
    codebase uses (queue, loop properties are forwarded).
    """

    def __init__(self, lane: Lane, policy: GroupPolicy) -> None:
        self._lane   = lane
        self._policy = policy
        self._buffer: list[InboundMessage] = []
        self._timeout_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Public interface (mirrors Lane for _SessionRouter)
    # ------------------------------------------------------------------

    async def push(self, msg: InboundMessage) -> bool:
        """Apply GroupPolicy and enqueue if appropriate. Returns False only on backpressure."""
        p    = self._policy
        text = msg.text.strip()

        if p.activation == ActivationMode.ALWAYS:
            # Respond to everything — no buffering, no stripping needed.
            return await self._lane.enqueue(msg)

        is_trigger = self._is_trigger(text, p)

        if not is_trigger:
            # Buffer the message as context for the next triggered turn.
            self._buffer.append(msg)
            if p.buffer_timeout_s > 0:
                self._reset_timeout()
            logger.debug(
                "GroupLane [%s]: buffered non-trigger from %s (%d in buffer)",
                self._lane.session_key, msg.author.username, len(self._buffer),
            )
            return True  # accepted (buffered, not dropped)

        # Trigger: strip, combine with buffer, enqueue.
        stripped_text = _gp_strip_trigger(text, p)
        trigger_msg   = _gp_replace_text(msg, stripped_text)
        combined      = self._flush_buffer(trigger_msg)

        logger.debug(
            "GroupLane [%s]: trigger from %s — flushing %d buffered + trigger",
            self._lane.session_key, msg.author.username, len(self._buffer),
        )
        return await self._lane.enqueue(combined)

    def reset(self) -> None:
        self._buffer.clear()
        self._cancel_timeout()
        self._lane.reset()

    async def stop(self) -> None:
        self._cancel_timeout()
        await self._lane.stop()

    # Forward lane properties so _SessionRouter can treat GroupLane like Lane.
    @property
    def queue(self):
        return self._lane.queue

    @property
    def loop(self):
        return self._lane.loop

    @property
    def session_key(self):
        return self._lane.session_key

    # ------------------------------------------------------------------
    # Policy mutation (for runtime /activation toggle)
    # ------------------------------------------------------------------

    def set_activation(self, mode: ActivationMode) -> None:
        """Replace the activation mode at runtime without clearing the buffer."""
        p = self._policy
        object.__setattr__(self, "_policy", GroupPolicy(
            activation=mode,
            trigger_prefix=p.trigger_prefix,
            bot_mxid=p.bot_mxid,
            bot_localpart=p.bot_localpart,
            buffer_timeout_s=p.buffer_timeout_s,
        ))
        logger.info("GroupLane [%s]: activation set to %s", self._lane.session_key, mode)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_trigger(self, text: str, p: GroupPolicy) -> bool:
        if p.activation == ActivationMode.PREFIX:
            return text.startswith(p.trigger_prefix)
        # MENTION mode: prefix OR @mention
        if text.startswith(p.trigger_prefix):
            return True
        if p.bot_mxid and p.bot_mxid in text:
            return True
        if p.bot_localpart and f"@{p.bot_localpart}" in text:
            return True
        return False

    def _flush_buffer(self, trigger: InboundMessage) -> InboundMessage:
        """Combine buffered non-trigger messages with the trigger message."""
        self._cancel_timeout()
        buffered = self._buffer[:]
        self._buffer.clear()

        if not buffered:
            return trigger

        lines = [f"[{m.author.username}]: {m.text}" for m in buffered]
        lines.append(trigger.text)
        return _gp_replace_text(trigger, "\n".join(lines))

    def _reset_timeout(self) -> None:
        self._cancel_timeout()
        self._timeout_task = asyncio.create_task(
            self._timeout_flush(),
            name=f"group-timeout:{self._lane.session_key}",
        )

    def _cancel_timeout(self) -> None:
        if self._timeout_task and not self._timeout_task.done():
            self._timeout_task.cancel()
        self._timeout_task = None

    async def _timeout_flush(self) -> None:
        """Flush buffered messages as a standalone turn after timeout."""
        try:
            await asyncio.sleep(self._policy.buffer_timeout_s)
            if not self._buffer:
                return
            # Use the last buffered message as the "trigger" (author/metadata)
            last = self._buffer[-1]
            combined = self._flush_buffer(last)
            logger.debug(
                "GroupLane [%s]: timeout flush — forwarding %d buffered messages",
                self._lane.session_key, len(self._buffer) + 1,
            )
            await self._lane.enqueue(combined)
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# _SessionRouter (internal)
# ---------------------------------------------------------------------------

class _SessionRouter:
    def __init__(self, config: Config, event_handler: EventHandler, router: "Router") -> None:
        self._config        = config
        self._event_handler = event_handler
        self._router        = router
        self._lanes: dict[SessionKey, Lane] = {}
        # version overrides set by next_session() before a lane exists
        self._version_overrides: dict[SessionKey, int] = {}

    async def route(self, msg: InboundMessage) -> bool:
        """Route a message to its lane. Returns False if the lane queue is full."""
        lane = self._get_or_create(msg.session_key, msg)
        # GroupLane.push() handles policy; plain Lane uses enqueue() directly.
        if isinstance(lane, GroupLane):
            accepted = await lane.push(msg)
        else:
            accepted = await lane.enqueue(msg)
        if accepted:
            logger.debug("Enqueued %s -> %s", msg.trace_id, msg.session_key)
        return accepted

    def _get_or_create(self, key: SessionKey, msg: InboundMessage) -> "Lane | GroupLane":
        if key not in self._lanes:
            version_override = self._version_overrides.pop(key, None)
            inner = Lane(
                session_key=key,
                config=self._config,
                event_handler=self._event_handler,
                router=self._router,
                version_override=version_override,
            )
            inner.start()

            if key.chat_type == ChatType.GROUP and msg.group_policy is not None:
                lane: Lane | GroupLane = GroupLane(inner, msg.group_policy)
                logger.info("Opened GroupLane %s (activation=%s)", key, msg.group_policy.activation)
            else:
                lane = inner
                logger.info("Opened lane %s", key)

            self._lanes[key] = lane
        return self._lanes[key]

    async def close_all(self) -> None:
        for lane in self._lanes.values():
            await lane.stop()  # Lane.stop() and GroupLane.stop() both exist
        self._lanes.clear()

    @property
    def active_sessions(self) -> list[SessionKey]:
        return list(self._lanes.keys())


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class Router:
    def __init__(self, config: Config) -> None:
        self._config             = config
        # Per-platform fallback handlers (cli, discord, matrix, cron, ...)
        self._platform_handlers: dict[str, EventHandler] = {}
        # Per-session handlers — take priority over platform handlers.
        # Registered for the lifetime of a single request (e.g. one SSE stream).
        self._session_handlers:  dict[SessionKey, EventHandler] = {}
        self._dm_platforms:      dict[SessionKey, str] = {}
        self._session_router     = _SessionRouter(
            config=config,
            event_handler=self._dispatch_event,
            router=self,
        )

    # ------------------------------------------------------------------
    # Handler registration
    # ------------------------------------------------------------------

    def register_platform_handler(self, platform: str, handler: EventHandler) -> None:
        """Register a fallback handler for all sessions on a platform."""
        self._platform_handlers[platform] = handler
        logger.info("Registered platform handler for '%s'", platform)

    # Alias kept for backward compat (cron module uses this name).
    def register_reply_handler(self, platform: str, handler: EventHandler) -> None:
        self.register_platform_handler(platform, handler)

    def register_session_handler(self, key: SessionKey, handler: EventHandler) -> None:
        """Register a per-session handler. Takes priority over platform handler."""
        self._session_handlers[key] = handler
        logger.debug("Registered session handler for %s", key)

    def unregister_session_handler(self, key: SessionKey) -> None:
        """Remove a per-session handler. No-op if not registered."""
        self._session_handlers.pop(key, None)
        logger.debug("Unregistered session handler for %s", key)

    # ------------------------------------------------------------------
    # Message push
    # ------------------------------------------------------------------

    async def push(self, msg: InboundMessage) -> bool:
        """Push a message. Returns False if the session lane queue is full."""
        if msg.session_key.chat_type.value == "dm":
            self._dm_platforms[msg.session_key] = msg.author.platform.value
        return await self._session_router.route(msg)

    # ------------------------------------------------------------------
    # Event dispatch
    # ------------------------------------------------------------------

    async def _dispatch_event(self, event: AgentEvent) -> None:
        sk = event.session_key

        # Per-session handler takes priority.
        handler = self._session_handlers.get(sk)
        if handler is not None:
            try:
                await handler(event)
            except Exception:
                logger.exception("Session handler failed for %s", event.trace_id)
            return

        # Fall back to platform handler.
        if sk.platform is not None:
            platform = sk.platform.value
        else:
            platform = self._dm_platforms.get(sk)

        if platform is None:
            logger.error("Cannot determine platform for session %s — dropping %s", sk, event.trace_id)
            return

        handler = self._platform_handlers.get(platform)
        if handler is None:
            logger.error("No handler for platform '%s' — dropping %s", platform, event.trace_id)
            return

        try:
            await handler(event)
        except Exception:
            logger.exception("Platform handler failed for %s", event.trace_id)

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def reset_session(self, key: SessionKey) -> None:
        """Hard reset — wipe session JSON from disk and clear context."""
        lane = self._session_router._lanes.get(key)
        if lane:
            lane.reset()  # GroupLane.reset() also clears the buffer
        else:
            logger.debug("reset_session: no lane for %s", key)

    def set_group_activation(self, key: SessionKey, mode: "str | ActivationMode") -> bool:
        """
        Change the activation mode for a group session at runtime.
        Useful for /activation commands in bridges.

        mode: 'mention' | 'prefix' | 'always'  (or ActivationMode enum)
        Returns True if the lane existed and was updated, False otherwise.
        """
        if isinstance(mode, str):
            try:
                mode = ActivationMode(mode)
            except ValueError:
                logger.error("set_group_activation: unknown mode %r", mode)
                return False
        lane = self._session_router._lanes.get(key)
        if isinstance(lane, GroupLane):
            lane.set_activation(mode)
            return True
        logger.debug("set_group_activation: no GroupLane for %s", key)
        return False

    def next_session(self, key: SessionKey) -> None:
        """Archive current session JSON and start a fresh version."""
        lane = self._session_router._lanes.get(key)
        if lane:
            lane.loop.next_session()
        else:
            # Lane doesn't exist yet — compute the next version from disk
            # and store it as an override so the lane starts there when created.
            safe_key = str(key).replace(":", "_")
            from pathlib import Path
            sessions_dir = Path("sessions") / safe_key
            existing = [
                int(p.stem) for p in sessions_dir.glob("*.json") if p.stem.isdigit()
            ] if sessions_dir.exists() else []
            next_version = max(existing, default=1) + 1
            self._session_router._version_overrides[key] = next_version
            logger.info("next_session: no lane for %s — set version override to v%d", key, next_version)

    async def shutdown(self) -> None:
        await self._session_router.close_all()

    @property
    def active_sessions(self) -> list[SessionKey]:
        return self._session_router.active_sessions
