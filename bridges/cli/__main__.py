"""
bridges/cli/__main__.py — Robust Interactive CLI bridge using Rich.
prompt_toolkit removed — no more patch_stdout fighting with Rich Live.
"""
from __future__ import annotations

import asyncio
import re
import time
import logging
from dataclasses import dataclass, field

import pyfiglet
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from contracts import (
    Platform, ContentType,
    SessionKey, UserIdentity, InboundMessage,
    AgentThinkingChunk, AgentTextChunk, AgentTextFinal, AgentToolCall, AgentToolResult, AgentError,
)

logger = logging.getLogger(__name__)

# --- LaTeX → Unicode Logic ---

try:
    from sympy import pretty
    from sympy.parsing.latex import parse_latex as _parse_latex
    # antlr4 is required at runtime — probe it now so we don't silently fall back later
    _parse_latex(r'x^2')
    _SYMPY = True
except Exception:
    _SYMPY = False
    logger.warning(
        "LaTeX -> unicode rendering disabled. "
        "Install sympy + antlr4-python3-runtime==4.11.0 to enable it."
    )

def _latex_to_unicode(latex: str) -> str:
    if not _SYMPY:
        return latex
    try:
        clean_latex = latex.strip().replace("**", "")
        expr = _parse_latex(clean_latex)
        return str(pretty(expr, use_unicode=True))
    except Exception:
        return latex

_BLOCK_MATH_DOLLARS  = re.compile(r'\$\$(.+?)\$\$', re.DOTALL)
_BLOCK_MATH_BRACKET  = re.compile(r'\\\[(.+?)\\\]', re.DOTALL)
_INLINE_MATH_DOLLARS = re.compile(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)', re.DOTALL)
_INLINE_MATH_PAREN   = re.compile(r'\\\((.+?)\\\)', re.DOTALL)
_FENCED_LATEX        = re.compile(r'^```[ \t]*(latex|math|tex)\s*\n(.+?)\n```[ \t]*$', re.DOTALL | re.MULTILINE | re.IGNORECASE)
_FENCED_CODE         = re.compile(r'^```[ \t]*(\w+)[ \t]*\n(.*?)\n```[ \t]*$', re.DOTALL | re.MULTILINE)
_FENCED_PLAIN        = re.compile(r'^```[ \t]*\n(.*?)\n```[ \t]*$', re.DOTALL | re.MULTILINE)

def _render_block(content: str) -> str:
    """Render a latex string as an indented unicode block."""
    rendered = _latex_to_unicode(content)
    indented = "\n".join("    " + line for line in rendered.splitlines())
    return f"\n{indented}\n"

def _preprocess_code_labels(text: str) -> str:
    """Prepend a dim language label above fenced code blocks since Rich drops it."""
    def _label(m: re.Match) -> str:
        lang = m.group(1)
        body = m.group(2)
        return f'*{lang}*\n```{lang}\n{body}\n```'
    return _FENCED_CODE.sub(_label, text)

def _preprocess_latex(text: str) -> str:
    """Replaces LaTeX spans with unicode before Markdown rendering.
    Handles fenced ```latex blocks and all four math delimiter styles.
    """
    def _fenced(m: re.Match) -> str:
        return _render_block(m.group(2))

    def _block(m: re.Match) -> str:
        return _render_block(m.group(1))

    def _inline(m: re.Match) -> str:
        return _latex_to_unicode(m.group(1))

    def _plain_fenced(m: re.Match) -> str:
        # No language tag — attempt LaTeX, fall back to plain code block
        content = m.group(1)
        rendered = _latex_to_unicode(content)
        if rendered != content:  # sympy changed it, treat as math
            return _render_block(content)
        return m.group(0)  # leave untouched for Rich to render

    text = _FENCED_LATEX.sub(_fenced, text)
    text = _FENCED_PLAIN.sub(_plain_fenced, text)
    text = _BLOCK_MATH_DOLLARS.sub(_block, text)
    text = _BLOCK_MATH_BRACKET.sub(_block, text)
    text = _INLINE_MATH_DOLLARS.sub(_inline, text)
    text = _INLINE_MATH_PAREN.sub(_inline, text)
    return text

# --- Theme & UI ---

@dataclass
class CLITheme:
    colors: dict[str, str] = field(default_factory=dict)
    text: dict[str, str] = field(default_factory=dict)

    def c(self, key: str) -> str:
        defaults = {
            "banner": "bright_cyan", "tagline": "bright_black", "border": "bright_black",
            "user_label": "green", "agent_label": "cyan", "thinking": "yellow",
            "tool_call": "bright_black", "tool_ok": "green", "tool_error": "red",
            "reset": "yellow", "error": "red",
        }
        return self.colors.get(key) or defaults.get(key, "")

    def t(self, key: str) -> str:
        defaults = {
            "name": "TinyCTX", "tagline": "AI Agent Framework",
            "user_label": "you", "agent_label": "agent", "bye_message": "Bye.",
        }
        return self.text.get(key) or defaults.get(key, "")


# --- The Bridge ---

class CLIBridge:
    def __init__(self, gateway, options: dict | None = None) -> None:
        self._gateway = gateway
        self._theme = CLITheme(
            colors=options.get("customcolors") or {} if options else {},
            text=options.get("customtext") or {} if options else {}
        )
        self._console = Console(highlight=False)
        self._reply_done = asyncio.Event()

        # State for the current turn
        self._current_content = ""
        self._is_thinking = False
        self._thinking_shown = False

    async def handle_event(self, event) -> None:
        c = self._theme.c
        t = self._theme.t

        if isinstance(event, AgentThinkingChunk):
            if not self._thinking_shown:
                self._thinking_shown = True
                # Simple inline indicator — no Live, no cursor fighting
                self._console.print(
                    f"[{c('thinking')}]{t('agent_label')}: ⠋ thinking...[/{c('thinking')}]",
                    end="\r"
                )

        elif isinstance(event, AgentTextChunk):
            self._is_thinking = False
            # Just buffer — rendering live conflicts with Rich's output
            self._current_content += event.text

        elif isinstance(event, AgentTextFinal):
            # Clear the thinking line if it was shown
            if self._thinking_shown:
                self._console.print(" " * 60, end="\r")

            final_text = (self._current_content + (event.text or "")).strip()
            processed = _preprocess_latex(final_text)
            processed = _preprocess_code_labels(processed)

            self._console.print(f"[{c('agent_label')}]{t('agent_label')}:[/{c('agent_label')}]")
            self._console.print(Markdown(processed))
            self._console.print()

            # Reset turn state
            self._current_content = ""
            self._is_thinking = False
            self._thinking_shown = False
            self._reply_done.set()

        elif isinstance(event, AgentToolCall):
            args_str = ", ".join(f"{k}={v!r}" for k, v in event.args.items())
            self._console.print(f"  [{c('tool_call')}]⟶  {event.tool_name}({args_str})[/{c('tool_call')}]")

        elif isinstance(event, AgentToolResult):
            status_color = c("tool_error") if event.is_error else c("tool_ok")
            icon = "✗" if event.is_error else "✓"
            preview = event.output[:100].replace("\n", " ") + ("..." if len(event.output) > 100 else "")
            self._console.print(f"  [{status_color}]{icon}  {event.tool_name}:[/{status_color}] ", end="")
            self._console.print(preview, markup=False, style="bright_black")

        elif isinstance(event, AgentError):
            self._console.print(f"\n[{c('error')}]error: {event.message}[/{c('error')}]\n")
            self._reply_done.set()

    async def _prompt(self, prompt_str: str) -> str:
        """Async input using a thread executor — no prompt_toolkit needed."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: input(prompt_str))

    async def run(self) -> None:
        self._gateway.register_platform_handler(Platform.CLI.value, self.handle_event)

        # Banner
        banner_text = Text()
        banner_text.append(pyfiglet.figlet_format(self._theme.t("name"), font="slant"), style=self._theme.c("banner"))
        banner_text.append(f"  {self._theme.t('tagline')}", style=self._theme.c("tagline"))
        self._console.print(Panel(banner_text, border_style=self._theme.c("border"), padding=(0, 2)))
        self._console.print(f"[{self._theme.c('border')}]  type a message · /reset · /next · exit[/{self._theme.c('border')}]\n")

        c = self._theme.c
        t = self._theme.t

        # Build a plain-text prompt string (Rich markup won't work in input())
        # Use ANSI directly for the colored prompt
        ANSI_RESET = "\033[0m"
        ANSI_GREEN = "\033[32m"
        prompt_str = f"{ANSI_GREEN}{t('user_label')}{ANSI_RESET}: "

        while True:
            try:
                text = await self._prompt(prompt_str)
                text = text.strip()
                if not text:
                    continue
                if text.lower() in {"exit", "quit"}:
                    break

                if text.startswith("/"):
                    if text.lower() == "/reset":
                        self._gateway.reset_session(CLI_SESSION)
                        self._console.print(f"[{c('reset')}]  ↺  context cleared[/{c('reset')}]")
                    elif text.lower() == "/next":
                        self._gateway.next_session(CLI_SESSION)
                        self._console.print(f"[{c('reset')}]  ↷  new session[/{c('reset')}]")
                    continue

                msg = InboundMessage(
                    session_key=CLI_SESSION, author=CLI_USER,
                    content_type=ContentType.TEXT, text=text,
                    message_id=str(time.time_ns()), timestamp=time.time(),
                )
                self._reply_done.clear()
                await self._gateway.push(msg)
                await self._reply_done.wait()

            except (KeyboardInterrupt, EOFError):
                break

        self._console.print(f"[{c('reset')}]{t('bye_message')}[/{c('reset')}]")


# Boilerplate Constants
CLI_USER_ID = "cli-owner"
CLI_USER = UserIdentity(platform=Platform.CLI, user_id=CLI_USER_ID, username="you")
CLI_SESSION = SessionKey.dm(CLI_USER_ID)


# --- Loader entry point ---

async def run(gateway) -> None:
    """Entry point called by main.py loader."""
    options = {}

    config = getattr(gateway, "_config", None)
    if config and hasattr(config, "bridges"):
        bridge_cfg = config.bridges.get("cli")
        if bridge_cfg:
            options = getattr(bridge_cfg, "options", {})

    bridge = CLIBridge(gateway, options=options)
    await bridge.run()


if __name__ == "__main__":
    from config import load as load_config, apply_logging
    from gateway import Gateway
    async def _main():
        cfg = load_config()
        apply_logging(cfg.logging)
        await run(Gateway(config=cfg))
    asyncio.run(_main())