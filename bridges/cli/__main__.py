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
    SessionKey, UserIdentity, InboundMessage,
    AgentThinkingChunk, AgentTextChunk, AgentTextFinal, AgentToolCall, AgentToolResult, AgentError,
)

logger = logging.getLogger(__name__)

CLI_USER_ID = "cli-owner"
CLI_USER    = UserIdentity(platform=Platform.CLI, user_id=CLI_USER_ID, username="you")
CLI_SESSION = SessionKey.dm(CLI_USER_ID)


class CLIBridge:
    def __init__(self, gateway) -> None:
        self._gateway    = gateway
        self._reply_done = asyncio.Event()
        self._streaming  = False  # True once we've printed "agent: " prefix
        self._thinking   = False  # True while reasoning_content tokens are arriving

    async def handle_event(self, event) -> None:
        if isinstance(event, AgentThinkingChunk):
            if not self._thinking:
                print("\n...Thinking...", end="", flush=True)
                self._thinking = True
            return

        if isinstance(event, AgentTextChunk):
            if self._thinking:
                # Clear the thinking indicator before first reply token
                print("\r                \r", end="", flush=True)
                self._thinking = False
            if not self._streaming:
                print("\nagent: ", end="", flush=True)
                self._streaming = True
            print(event.text, end="", flush=True)

        elif isinstance(event, AgentTextFinal):
            self._thinking = False
            if self._streaming:
                if event.text:
                    # Non-streaming fallback: full text arrived at once.
                    print(f"\nagent: {event.text}")
                else:
                    print()  # close the streamed line
            else:
                print(f"\nagent: {event.text}")
            print()
            self._streaming = False
            self._reply_done.set()

        elif isinstance(event, AgentToolCall):
            if self._thinking:
                print("\r                \r", end="", flush=True)
                self._thinking = False
            args_str = ", ".join(f"{k}={v!r}" for k, v in event.args.items())
            print(f"\n[tool → {event.tool_name}({args_str})]")

        elif isinstance(event, AgentToolResult):
            status = "error" if event.is_error else "ok"
            preview = event.output[:120].replace("\n", " ")
            if len(event.output) > 120:
                preview += "..."
            print(f"[tool ← {event.tool_name} ({status}): {preview}]")

        elif isinstance(event, AgentError):
            print(f"\nagent: {event.message}\n")
            self._streaming = False
            self._reply_done.set()

    async def run(self) -> None:
        self._gateway.register_platform_handler(Platform.CLI.value, self.handle_event)
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
                    self._streaming = False
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