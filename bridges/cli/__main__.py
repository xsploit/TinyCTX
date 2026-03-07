"""
bridges/cli/__main__.py — Interactive CLI bridge.

Exposes run(gateway) for main.py loader.
Can still be run standalone: python -m bridges.cli

Streaming: partial chunks (is_partial=True) are printed inline without a
newline as they arrive. The closing chunk (is_partial=False) prints the
final newline and unblocks the input prompt.
"""
from __future__ import annotations

import asyncio
import sys
import time
import logging

from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout

from contracts import (
    Platform, ContentType,
    SessionKey, UserIdentity, InboundMessage, OutboundReply,
)

logger = logging.getLogger(__name__)

CLI_USER_ID = "cli-owner"
CLI_USER    = UserIdentity(platform=Platform.CLI, user_id=CLI_USER_ID, username="you")
CLI_SESSION = SessionKey.dm(CLI_USER_ID)


class CLIBridge:
    def __init__(self, gateway) -> None:
        self._gateway    = gateway
        self._reply_done = asyncio.Event()
        self._started    = False  # True once we've printed "agent: " prefix

    async def handle_reply(self, reply: OutboundReply) -> None:
        if reply.is_partial:
            # First partial chunk of a new reply — print the "agent: " prefix.
            if not self._started:
                # Print prefix without newline; flush immediately so it appears
                # before the first word streams in.
                print("\nagent: ", end="", flush=True)
                self._started = True
            # Stream the token directly to stdout, no newline.
            print(reply.text, end="", flush=True)
        else:
            # Final chunk — close the line.
            if self._started:
                # Streaming reply: we already printed the prefix + tokens,
                # just add the closing newline.
                if reply.text:
                    # Non-streaming fallback path: text arrived all at once.
                    print(f"\nagent: {reply.text}")
                else:
                    print()  # close the streamed line
            else:
                # No partials arrived (e.g. error path) — print the whole thing.
                print(f"\nagent: {reply.text}")

            print()  # blank line after each reply for readability
            self._started = False
            self._reply_done.set()

    async def run(self) -> None:
        self._gateway.register_reply_handler(Platform.CLI.value, self.handle_reply)
        print("CLI bridge ready. Type a message, Ctrl-C or 'exit' to quit.\n")

        with patch_stdout():
            while True:
                try:
                    text = await PromptSession().prompt_async("you: ")
                except (KeyboardInterrupt, EOFError):
                    print("\nBye.")
                    break

                text = text.strip()
                if not text:
                    continue
                if text.lower() in {"exit", "quit"}:
                    print("Bye.")
                    break
                if text.lower() == "/reset":
                    self._gateway.reset_session(CLI_SESSION)
                    self._started = False
                    print("\n[context cleared]\n")
                    continue

                msg = InboundMessage(
                    session_key=CLI_SESSION,
                    author=CLI_USER,
                    content_type=ContentType.TEXT,
                    text=text,
                    message_id=str(time.time_ns()),
                    timestamp=time.time(),
                )

                self._reply_done.clear()
                await self._gateway.push(msg)
                await self._reply_done.wait()


async def run(gateway) -> None:
    """Entry point called by main.py loader."""
    bridge = CLIBridge(gateway)
    await bridge.run()


if __name__ == "__main__":
    import asyncio
    from config import load as load_config, apply_logging
    from gateway import Gateway

    async def _standalone():
        cfg = load_config()
        apply_logging(cfg.logging)
        await run(Gateway(config=cfg))

    asyncio.run(_standalone())