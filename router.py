"""
router.py — Async session router and lane queues.
No LLM calls, no tools, no memory. Pure routing and async primitives.
Imports only from contracts.py, agent.py, config.py, and stdlib.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Callable, Awaitable

from contracts import InboundMessage, AgentEvent, SessionKey, ChatType
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
        lane = self._get_or_create(msg.session_key)
        accepted = await lane.enqueue(msg)
        if accepted:
            logger.debug("Enqueued %s -> %s", msg.trace_id, msg.session_key)
        return accepted

    def _get_or_create(self, key: SessionKey) -> Lane:
        if key not in self._lanes:
            version_override = self._version_overrides.pop(key, None)
            lane = Lane(
                session_key=key,
                config=self._config,
                event_handler=self._event_handler,
                router=self._router,
                version_override=version_override,
            )
            lane.start()
            self._lanes[key] = lane
            logger.info("Opened lane %s", key)
        return self._lanes[key]

    async def close_all(self) -> None:
        for lane in self._lanes.values():
            await lane.stop()
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
            lane.reset()
        else:
            logger.debug("reset_session: no lane for %s", key)

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
