from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import yaml

from prompt_toolkit.data_structures import Point
from prompt_toolkit.document import Document
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from prompt_toolkit.selection import SelectionState, SelectionType

from bridges.cli.__main__ import (
    CLIBridge,
    _DimToolLineProcessor,
    _MarkdownLineProcessor,
    _SlashCommandCompleter,
)
from contracts import AgentError, AgentToolResult
from config import (
    BridgeConfig,
    Config,
    LLMRoutingConfig,
    LoggingConfig,
    ModelConfig,
    WorkspaceConfig,
)
from db import ConversationDB
from main import _startup_log_level


def _make_config(
    tmp_path: Path,
    *,
    logging_level: str = "INFO",
    cli_options: dict | None = None,
    extra: dict | None = None,
) -> Config:
    return Config(
        models={
            "main": ModelConfig(
                base_url="http://localhost:8080/v1",
                model="llama3",
                api_key_env="N/A",
            )
        },
        llm=LLMRoutingConfig(primary="main"),
        workspace=WorkspaceConfig(path=tmp_path),
        logging=LoggingConfig(level=logging_level),
        bridges={"cli": BridgeConfig(enabled=True, options=cli_options or {})},
        extra=extra or {},
    )


def test_startup_log_level_defaults_to_warning_for_cli(tmp_path):
    cfg = _make_config(tmp_path, logging_level="INFO")
    assert _startup_log_level(cfg) == logging.WARNING


def test_startup_log_level_can_keep_info_when_quiet_startup_disabled(tmp_path):
    cfg = _make_config(tmp_path, logging_level="INFO", cli_options={"quiet_startup": False})
    assert _startup_log_level(cfg) == logging.INFO


def test_cli_runtime_log_level_defaults_to_warning(tmp_path):
    cfg = _make_config(tmp_path, logging_level="INFO")
    bridge = CLIBridge(SimpleNamespace(_config=cfg), options={})
    assert bridge._resolve_runtime_log_level() == logging.WARNING


def test_cli_runtime_log_level_can_inherit_global_level(tmp_path):
    cfg = _make_config(tmp_path, logging_level="INFO")
    bridge = CLIBridge(SimpleNamespace(_config=cfg), options={"log_level": "inherit"})
    assert bridge._resolve_runtime_log_level() == logging.INFO


def test_cli_startup_summary_is_compact_and_informative(tmp_path):
    cfg = _make_config(
        tmp_path,
        extra={
            "memory": {"embedding_model": "embed"},
            "heartbeat": {"every_minutes": 15},
            "mcp": {"servers": {"github": {"command": "uvx"}}},
        },
    )
    bridge = CLIBridge(SimpleNamespace(_config=cfg), options={})
    summary = bridge._startup_summary(logging.WARNING).plain
    assert "workspace" in summary
    assert "model llama3" in summary
    assert "memory embed" in summary
    assert "heartbeat 15m" in summary
    assert "mcp 1" in summary
    assert "logs warning+" in summary


def test_cli_welcome_screen_uses_tinyctx_ascii_logo_and_shortcuts(tmp_path):
    cfg = _make_config(tmp_path)
    bridge = CLIBridge(SimpleNamespace(_config=cfg), options={})
    welcome = bridge._compose_welcome_text(logging.WARNING, width=100)
    assert "████████╗██╗███╗   ██╗██╗   ██╗" in welcome
    assert "Agent Framework" in welcome
    assert "cwd " in welcome
    first_left_line = bridge._welcome_lines(logging.WARNING, width=60)[0]
    assert first_left_line.startswith(" ")
    wide_first_line = bridge._compose_welcome_text(logging.WARNING, width=140).splitlines()[0]
    assert len(wide_first_line) - len(wide_first_line.lstrip(" ")) > 20


def test_cli_footer_tracks_working_status(tmp_path):
    cfg = _make_config(tmp_path)
    bridge = CLIBridge(SimpleNamespace(_config=cfg), options={})
    bridge._set_status("web_search")
    footer = bridge._footer_text()
    assert "working web_search" in footer
    assert "Enter send" not in footer


def test_cli_output_wraps_while_input_stays_single_line(tmp_path):
    cfg = _make_config(tmp_path)
    bridge = CLIBridge(SimpleNamespace(_config=cfg), options={})
    with patch("bridges.cli.__main__.Application", return_value=SimpleNamespace()) as app_cls:
        app = bridge._build_application()
    assert bridge._output_area is not None
    assert bridge._input_area is not None
    assert bridge._output_area.wrap_lines is True
    assert bridge._output_area.control.focusable() is True
    assert bridge._output_area.control.focus_on_click() is True
    assert bridge._input_area.buffer.multiline() is False
    assert app_cls.called
    assert app is not None


def test_cli_mouse_capture_defaults_on(tmp_path):
    cfg = _make_config(tmp_path)
    bridge = CLIBridge(SimpleNamespace(_config=cfg), options={})
    with patch("bridges.cli.__main__.Application", return_value=SimpleNamespace()) as app_cls:
        bridge._build_application()

    mouse_support = app_cls.call_args.kwargs["mouse_support"]
    assert mouse_support() is True


def test_cli_page_scroll_moves_transcript_window(tmp_path):
    cfg = _make_config(tmp_path)
    bridge = CLIBridge(SimpleNamespace(_config=cfg), options={})
    with patch("bridges.cli.__main__.Application", return_value=SimpleNamespace()):
        bridge._build_application()

    assert bridge._output_area is not None
    bridge._application = SimpleNamespace(
        output=SimpleNamespace(get_size=lambda: SimpleNamespace(columns=120, rows=30)),
        invalidate=MagicMock(),
    )
    bridge._output_area.buffer.set_document(
        Document(
            text="\n".join(f"line {i}" for i in range(200)),
            cursor_position=0,
        ),
        bypass_readonly=True,
    )

    bridge._scroll_output_pages(1)
    assert bridge._output_area.buffer.cursor_position > 0

    bridge._scroll_output_pages(-1)
    assert bridge._output_area.buffer.cursor_position == 0


def test_cli_refresh_output_preserves_scrolled_position(tmp_path):
    cfg = _make_config(tmp_path)
    bridge = CLIBridge(SimpleNamespace(_config=cfg), options={})
    with patch("bridges.cli.__main__.Application", return_value=SimpleNamespace()):
        bridge._build_application()

    assert bridge._output_area is not None
    bridge._transcript_blocks = [f"line {i}" for i in range(200)]
    bridge._refresh_output(logging.WARNING)
    bridge._scroll_output_pages(-3)
    scrolled_cursor = bridge._output_area.buffer.cursor_position
    assert scrolled_cursor < len(bridge._output_area.buffer.text)

    bridge._append_block("new line after scroll")
    bridge._refresh_output(logging.WARNING)

    assert bridge._output_area.buffer.cursor_position == scrolled_cursor


def test_cli_drag_selection_can_autoscroll_output(tmp_path):
    cfg = _make_config(tmp_path)
    bridge = CLIBridge(SimpleNamespace(_config=cfg), options={})
    with patch("bridges.cli.__main__.Application", return_value=SimpleNamespace()):
        bridge._build_application()

    async def _run() -> None:
        assert bridge._output_area is not None
        bridge._loop = asyncio.get_running_loop()
        bridge._application = SimpleNamespace(
            layout=SimpleNamespace(focus=MagicMock()),
            invalidate=MagicMock(),
        )

        original_mouse_handler = MagicMock(return_value=None)
        bridge._output_area.control.mouse_handler = original_mouse_handler
        bridge._wrap_mouse_handler_for_paste(bridge._output_area)
        bridge._output_area.window.render_info = SimpleNamespace(
            displayed_lines=[object()] * 20,
            top_visible=False,
            bottom_visible=False,
        )
        bridge._output_area.window._scroll_down = MagicMock()
        bridge._output_area.window._scroll_up = MagicMock()

        mouse_down = SimpleNamespace(
            event_type=MouseEventType.MOUSE_DOWN,
            button=MouseButton.LEFT,
            position=Point(x=0, y=10),
        )
        mouse_move = SimpleNamespace(
            event_type=MouseEventType.MOUSE_MOVE,
            button=MouseButton.LEFT,
            position=Point(x=0, y=19),
        )

        bridge._output_area.control.mouse_handler(mouse_down)
        bridge._output_area.control.mouse_handler(mouse_move)
        await asyncio.sleep(0.12)
        bridge._stop_output_drag_tracking()
        await asyncio.sleep(0)

        assert bridge._output_area.window._scroll_down.call_count >= 2
        assert original_mouse_handler.call_count >= 3

    asyncio.run(_run())


def test_cli_style_uses_black_background_and_red_banner(tmp_path):
    cfg = _make_config(tmp_path)
    bridge = CLIBridge(SimpleNamespace(_config=cfg), options={})
    style = bridge._style().style_rules
    assert ("output-area", "#d7d7d7 bg:#000000") in style
    assert ("input-area", "#f5f5f5 bg:#000000") in style
    assert ("banner", "bold #ff3b30 bg:#000000") in style
    assert ("tool-dim", "#7f7f7f bg:#000000") in style
    assert ("md.heading.marker", "bold #ff3b30 bg:#000000") in style
    assert ("md.inline-code", "#ffb86c bg:#111111") in style
    first_fragment = bridge._welcome_fragments()[0]
    assert first_fragment[0] == "class:banner"


def test_cli_tool_lines_are_compact(tmp_path):
    cfg = _make_config(tmp_path)
    bridge = CLIBridge(SimpleNamespace(_config=cfg), options={})
    assert bridge._tool_call_line("web_search", {"query": "NHL scores today"}) == "[tool web_search NHL scores today]"
    assert bridge._tool_call_line("browse_url", {"url": "https://example.com", "mode": "text"}) == "[tool browse_url https://example.com]"
    assert bridge._tool_result_line("web_search", "Search results for NHL scores today", False) == "[ok web_search Search results for NHL scores today]"


def test_cli_dims_tool_prefix_lines():
    processor = _DimToolLineProcessor()
    transformed = processor.apply_transformation(
        SimpleNamespace(fragments=[("", "tool web_search NHL scores today")])
    )
    assert transformed.fragments == [("class:tool-dim", "tool web_search NHL scores today")]


def test_cli_markdown_processor_styles_headings():
    processor = _MarkdownLineProcessor()
    transformed = processor.apply_transformation(
        SimpleNamespace(
            fragments=[("", "## Heading")],
            lineno=0,
            document=SimpleNamespace(lines=["## Heading"]),
        )
    )
    assert transformed.fragments == [
        ("", ""),
        ("class:md.heading.marker", "##"),
        ("class:md.heading", " Heading"),
    ]


def test_cli_markdown_processor_styles_inline_code_bold_and_links():
    processor = _MarkdownLineProcessor()
    transformed = processor.apply_transformation(
        SimpleNamespace(
            fragments=[("", "Use `code`, **bold**, and [docs](https://example.com).")],
            lineno=0,
            document=SimpleNamespace(lines=["Use `code`, **bold**, and [docs](https://example.com)."]),
        )
    )
    styles = [style for style, _ in transformed.fragments if style]
    assert "class:md.inline-code" in styles
    assert "class:md.bold" in styles
    assert "class:md.link" in styles


def test_cli_markdown_processor_styles_fenced_code_blocks():
    processor = _MarkdownLineProcessor()
    transformed = processor.apply_transformation(
        SimpleNamespace(
            fragments=[("", "print('x')")],
            lineno=1,
            document=SimpleNamespace(lines=["```python", "print('x')", "```"]),
        )
    )
    assert transformed.fragments == [("class:md.code", "print('x')")]


def test_cli_wraps_transcript_by_words(tmp_path):
    cfg = _make_config(tmp_path)
    bridge = CLIBridge(SimpleNamespace(_config=cfg), options={})
    wrapped = bridge._wrap_text_line(
        "Based on the latest NHL standings, the Anaheim Ducks are first in the Pacific Division.",
        40,
    )
    assert "Pacific" in wrapped
    assert "Pacif\nic" not in wrapped


def test_slash_command_completer_matches_prefix():
    completions = list(
        _SlashCommandCompleter().get_completions(
            Document(text="/se", cursor_position=3),
            None,
        )
    )
    assert [completion.display_text for completion in completions] == ["/settings"]
    assert [completion.text for completion in completions] == ["ttings"]


def test_slash_command_completer_supports_multiword_command():
    completions = list(
        _SlashCommandCompleter().get_completions(
            Document(text="/debug h", cursor_position=8),
            None,
        )
    )
    assert [completion.display_text for completion in completions] == ["/debug heartbeat"]
    assert [completion.text for completion in completions] == ["eartbeat"]


def test_slash_command_completer_supports_copy_tool_command():
    completions = list(
        _SlashCommandCompleter().get_completions(
            Document(text="/copy last-t", cursor_position=12),
            None,
        )
    )
    assert [completion.display_text for completion in completions] == [
        "/copy last-tool",
        "/copy last-tool-call",
        "/copy last-tool-result",
    ]
    assert [completion.text for completion in completions] == [
        "ool",
        "ool-call",
        "ool-result",
    ]


def test_slash_command_completer_supports_mouse_command():
    completions = list(
        _SlashCommandCompleter().get_completions(
            Document(text="/mo", cursor_position=3),
            None,
        )
    )
    assert [completion.display_text for completion in completions] == [
        "/mouse",
        "/mouse off",
        "/mouse on",
    ]


def test_settings_command_opens_menu(tmp_path):
    cfg = _make_config(tmp_path)
    bridge = CLIBridge(SimpleNamespace(_config=cfg), options={})
    asyncio.run(bridge._handle_command("/settings"))
    assert bridge._settings_open() is True
    assert bridge._settings_menu()[0] == "Settings"
    assert bridge._footer_text() == "working settings | mouse"


def test_settings_navigation_enters_submenu_and_applies_choice(tmp_path):
    cfg = _make_config(tmp_path)
    bridge = CLIBridge(SimpleNamespace(_config=cfg), options={})
    bridge._open_settings()
    bridge._move_settings(2)
    bridge._activate_settings_selection()
    assert bridge._settings_path[-1] == "behavior"
    bridge._move_settings(2)
    bridge._activate_settings_selection()
    assert bridge._settings_path[-1] == "log_level"
    bridge._move_settings(3)
    bridge._activate_settings_selection()
    assert bridge._settings_path[-1] == "behavior"
    assert bridge._options["log_level"] == "debug"


def test_settings_round_trips_menu_updates_runtime_config(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
models:
  main:
    base_url: http://localhost:8080/v1
    model: llama3
    api_key_env: N/A
llm:
  primary: main
max_tool_cycles: 20
bridges:
  cli:
    enabled: true
    options: {}
""".strip(),
        encoding="utf-8",
    )
    cfg = _make_config(tmp_path)
    cfg.max_tool_cycles = 20
    setattr(cfg, "_source_path", cfg_path)
    bridge = CLIBridge(SimpleNamespace(_config=cfg), options={})
    bridge._open_settings()
    bridge._move_settings(2)
    bridge._activate_settings_selection()
    assert bridge._settings_path[-1] == "behavior"
    bridge._activate_settings_selection()
    assert bridge._settings_path[-1] == "round_trips"
    bridge._move_settings(2)
    bridge._activate_settings_selection()
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert raw["max_tool_cycles"] == 30
    assert cfg.max_tool_cycles == 30


def test_settings_toggle_persists_cli_option(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
models:
  main:
    base_url: http://localhost:8080/v1
    model: llama3
    api_key_env: N/A
llm:
  primary: main
bridges:
  cli:
    enabled: true
    options:
      compact_tools: true
""".strip(),
        encoding="utf-8",
    )
    cfg = _make_config(tmp_path)
    setattr(cfg, "_source_path", cfg_path)
    bridge = CLIBridge(SimpleNamespace(_config=cfg), options={"compact_tools": True})
    bridge._apply_cli_option("compact_tools", False)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert raw["bridges"]["cli"]["options"]["compact_tools"] is False


def test_mouse_command_toggles_runtime_option(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
models:
  main:
    base_url: http://localhost:8080/v1
    model: llama3
    api_key_env: N/A
llm:
  primary: main
bridges:
  cli:
    enabled: true
    options:
      mouse_capture: true
""".strip(),
        encoding="utf-8",
    )
    cfg = _make_config(tmp_path)
    setattr(cfg, "_source_path", cfg_path)
    bridge = CLIBridge(SimpleNamespace(_config=cfg), options={"mouse_capture": True})

    asyncio.run(bridge._handle_command("/mouse"))
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert raw["bridges"]["cli"]["options"]["mouse_capture"] is False
    assert bridge._options["mouse_capture"] is False

    asyncio.run(bridge._handle_command("/mouse on"))
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert raw["bridges"]["cli"]["options"]["mouse_capture"] is True
    assert bridge._options["mouse_capture"] is True


def test_mouse_command_updates_footer_without_transcript_spam(tmp_path):
    cfg = _make_config(tmp_path)
    bridge = CLIBridge(SimpleNamespace(_config=cfg), options={})

    assert "working ready | mouse" in bridge._footer_text()

    asyncio.run(bridge._handle_command("/mouse on"))
    assert bridge._transcript_blocks == []
    assert "working ready | mouse" in bridge._footer_text()

    asyncio.run(bridge._handle_command("/mouse off"))
    assert bridge._transcript_blocks == []
    assert "working ready | selection" in bridge._footer_text()


def test_settings_appearance_menu_shows_mouse_capture(tmp_path):
    cfg = _make_config(tmp_path)
    bridge = CLIBridge(SimpleNamespace(_config=cfg), options={})
    bridge._settings_path = ["root", "appearance"]
    bridge._settings_selected = [0, 0]
    lines = "".join(fragment[1] for fragment in bridge._settings_fragments())
    assert "Mouse capture" in lines


def test_settings_root_contains_session_submenu(tmp_path):
    cfg = _make_config(tmp_path)
    bridge = CLIBridge(SimpleNamespace(_config=cfg), options={})
    bridge._open_settings()
    lines = "".join(fragment[1] for fragment in bridge._settings_fragments())
    assert "Providers" in lines
    assert "Appearance" in lines
    assert "Behavior" in lines
    assert "Session" in lines


def test_settings_behavior_menu_shows_round_trips_value(tmp_path):
    cfg = _make_config(tmp_path)
    cfg.max_tool_cycles = 20
    bridge = CLIBridge(SimpleNamespace(_config=cfg), options={})
    bridge._settings_path = ["root", "behavior"]
    bridge._settings_selected = [0, 0]
    lines = "".join(fragment[1] for fragment in bridge._settings_fragments())
    assert "Agent round trips" in lines
    assert "20" in lines


def test_settings_behavior_menu_shows_compaction_value(tmp_path):
    cfg = _make_config(tmp_path)
    bridge = CLIBridge(SimpleNamespace(_config=cfg), options={})
    bridge._settings_path = ["root", "behavior"]
    bridge._settings_selected = [0, 0]
    lines = "".join(fragment[1] for fragment in bridge._settings_fragments())
    assert "Context compaction" in lines
    assert "on" in lines


def test_settings_compaction_menu_updates_runtime_config(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
models:
  main:
    base_url: http://localhost:8080/v1
    model: llama3
    api_key_env: N/A
llm:
  primary: main
compaction:
  enabled: true
  trigger_pct: 0.9
  keep_last_units: 4
bridges:
  cli:
    enabled: true
    options: {}
""".strip(),
        encoding="utf-8",
    )
    cfg = _make_config(tmp_path)
    setattr(cfg, "_source_path", cfg_path)
    bridge = CLIBridge(SimpleNamespace(_config=cfg), options={})

    bridge._settings_path = ["root", "behavior", "compaction"]
    bridge._settings_selected = [0, 0, 0]
    bridge._activate_settings_selection()

    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert raw["compaction"]["enabled"] is False
    assert cfg.compaction.enabled is False


def test_settings_compaction_trigger_menu_updates_runtime_config(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
models:
  main:
    base_url: http://localhost:8080/v1
    model: llama3
    api_key_env: N/A
llm:
  primary: main
compaction:
  enabled: true
  trigger_pct: 0.9
  keep_last_units: 4
bridges:
  cli:
    enabled: true
    options: {}
""".strip(),
        encoding="utf-8",
    )
    cfg = _make_config(tmp_path)
    setattr(cfg, "_source_path", cfg_path)
    bridge = CLIBridge(SimpleNamespace(_config=cfg), options={})

    bridge._settings_path = ["root", "behavior", "compaction", "compaction_trigger"]
    bridge._settings_selected = [0, 0, 0, 2]
    bridge._activate_settings_selection()

    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert raw["compaction"]["trigger_pct"] == 1.0
    assert cfg.compaction.trigger_pct == 1.0
    assert bridge._settings_path[-1] == "compaction"


def test_settings_providers_menu_shows_active_profile_context(tmp_path):
    cfg = _make_config(tmp_path)
    bridge = CLIBridge(SimpleNamespace(_config=cfg), options={})
    bridge._settings_path = ["root", "providers"]
    bridge._settings_selected = [0, 0]
    lines = "".join(fragment[1] for fragment in bridge._settings_fragments())
    assert "active profile main" in lines
    assert "Manage active profile" in lines


def test_settings_can_create_provider_preset_profile(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
models:
  main:
    base_url: http://localhost:8080/v1
    model: llama3
    api_key_env: N/A
llm:
  primary: main
bridges:
  cli:
    enabled: true
    options: {}
""".strip(),
        encoding="utf-8",
    )
    cfg = _make_config(tmp_path)
    cfg.models["ollama"] = ModelConfig(
        base_url="http://localhost:11434/v1",
        model="llama3.1",
        api_key_env="N/A",
    )
    setattr(cfg, "_source_path", cfg_path)
    bridge = CLIBridge(SimpleNamespace(_config=cfg), options={})
    bridge._settings_path = ["root", "providers", "providers_add"]
    bridge._settings_selected = [0, 0, 0]
    bridge._activate_settings_selection()
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert raw["models"]["openai"]["base_url"] == "https://api.openai.com/v1"
    assert cfg.models["openai"].base_url == "https://api.openai.com/v1"
    assert bridge._settings_path[-1] == "provider_profile:openai"


def test_settings_can_set_primary_profile_from_provider_menu(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
models:
  main:
    base_url: http://localhost:8080/v1
    model: llama3
    api_key_env: N/A
  ollama:
    base_url: http://localhost:11434/v1
    model: llama3.1
    api_key_env: N/A
llm:
  primary: main
bridges:
  cli:
    enabled: true
    options: {}
""".strip(),
        encoding="utf-8",
    )
    cfg = _make_config(tmp_path)
    cfg.models["ollama"] = ModelConfig(
        base_url="http://localhost:11434/v1",
        model="llama3.1",
        api_key_env="N/A",
    )
    setattr(cfg, "_source_path", cfg_path)
    bridge = CLIBridge(SimpleNamespace(_config=cfg), options={})
    bridge._settings_path = ["root", "providers", "providers_primary"]
    bridge._settings_selected = [0, 0, 1]
    bridge._activate_settings_selection()
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert raw["llm"]["primary"] == "ollama"
    assert cfg.llm.primary == "ollama"


def test_tab_completion_fills_unique_slash_command(tmp_path):
    cfg = _make_config(tmp_path)
    bridge = CLIBridge(SimpleNamespace(_config=cfg), options={})
    with patch("bridges.cli.__main__.Application", return_value=SimpleNamespace()):
        bridge._build_application()
    assert bridge._input_area is not None
    assert isinstance(bridge._input_area.completer, _SlashCommandCompleter)
    bridge._input_area.buffer.document = Document(text="/set", cursor_position=4)
    bridge._complete_input()
    assert bridge._input_area.text == "/settings"


def test_cli_capture_root_logs_routes_warnings_to_transcript(tmp_path):
    cfg = _make_config(tmp_path)
    bridge = CLIBridge(SimpleNamespace(_config=cfg), options={})
    root = logging.getLogger()
    previous_handlers = list(root.handlers)
    previous_level = root.level

    with bridge._capture_root_logs(logging.INFO):
        logging.info("hidden info")
        logging.warning("keep warning")

    assert "warn keep warning" in bridge._transcript_blocks
    assert all("hidden info" not in block for block in bridge._transcript_blocks)
    assert root.handlers == previous_handlers
    assert root.level == previous_level


def test_large_paste_collapses_to_reference_and_expands_for_submit(tmp_path):
    cfg = _make_config(tmp_path)
    bridge = CLIBridge(SimpleNamespace(_config=cfg), options={})
    with patch("bridges.cli.__main__.Application", return_value=SimpleNamespace()):
        bridge._build_application()
    assert bridge._input_area is not None

    pasted = "line one\nline two\nline three"
    bridge._handle_paste(pasted)

    assert bridge._input_area.text == "[Pasted text #1, 28 chars]"
    assert bridge._expand_pasted_text_refs(bridge._input_area.text) == pasted


def test_short_paste_also_uses_reference(tmp_path):
    cfg = _make_config(tmp_path)
    bridge = CLIBridge(SimpleNamespace(_config=cfg), options={})
    with patch("bridges.cli.__main__.Application", return_value=SimpleNamespace()):
        bridge._build_application()

    bridge._handle_paste("hello")
    assert bridge._input_area is not None
    assert bridge._input_area.text == "[Pasted text #1, 5 chars]"
    assert bridge._expand_pasted_text_refs(bridge._input_area.text) == "hello"


def test_submit_text_renders_placeholder_but_sends_expanded_paste(tmp_path):
    cfg = _make_config(tmp_path)

    pushed = {}

    async def _push(msg):
        pushed["text"] = msg.text
        bridge._reply_done.set()

    gateway = SimpleNamespace(_config=cfg, push=_push)
    bridge = CLIBridge(gateway, options={})
    bridge._cursor = "cursor-1"

    asyncio.run(
        bridge._submit_text(
            "[Pasted text #1, 10 chars]",
            pasted_texts={1: "alpha\nbeta"},
        )
    )

    assert pushed["text"] == "alpha\nbeta"
    assert bridge._transcript_blocks[-1] == "› [Pasted text #1, 10 chars]"


def test_submit_from_buffer_reports_running_generation(tmp_path):
    cfg = _make_config(tmp_path)
    bridge = CLIBridge(SimpleNamespace(_config=cfg), options={})
    with patch("bridges.cli.__main__.Application", return_value=SimpleNamespace()):
        bridge._build_application()

    assert bridge._input_area is not None
    bridge._input_area.text = "hello"
    bridge._send_task = MagicMock()
    bridge._send_task.done.return_value = False

    bridge._submit_from_buffer()

    assert bridge._input_area.text == "hello"
    assert bridge._transcript_blocks[-1] == "[generation already running — press Esc to abort or wait for the current reply]"
    assert bridge._footer_text() == "working busy | mouse"


def test_abort_active_generation_calls_gateway(tmp_path):
    cfg = _make_config(tmp_path)
    gateway = SimpleNamespace(_config=cfg, abort_generation=MagicMock(return_value=True))
    bridge = CLIBridge(gateway, options={})
    bridge._cursor = "cursor-1"
    bridge._send_task = MagicMock()
    bridge._send_task.done.return_value = False

    assert bridge._abort_active_generation() is True
    gateway.abort_generation.assert_called_once_with("cursor-1")
    assert bridge._footer_text() == "working aborting | mouse"


def test_copy_primary_text_prefers_selected_output(tmp_path):
    cfg = _make_config(tmp_path)
    bridge = CLIBridge(SimpleNamespace(_config=cfg), options={})
    with patch("bridges.cli.__main__.Application", return_value=SimpleNamespace()):
        bridge._build_application()

    assert bridge._output_area is not None
    bridge._output_area.buffer.set_document(
        Document(
            text="picked output",
            cursor_position=len("picked output"),
        ),
        bypass_readonly=True,
    )
    bridge._output_area.buffer.selection_state = SelectionState(
        original_cursor_position=0,
        type=SelectionType.CHARACTERS,
    )
    with patch.object(bridge, "_write_clipboard_text", return_value=True) as copy_mock:
        assert bridge._copy_primary_text() is True

    copy_mock.assert_called_once_with("picked output")


def test_copy_primary_text_falls_back_to_full_transcript(tmp_path):
    cfg = _make_config(tmp_path)
    bridge = CLIBridge(SimpleNamespace(_config=cfg), options={})
    with patch("bridges.cli.__main__.Application", return_value=SimpleNamespace()):
        bridge._build_application()

    bridge._transcript_blocks = ["line one", "line two"]
    bridge._refresh_output(logging.WARNING)

    with patch.object(bridge, "_write_clipboard_text", return_value=True) as copy_mock:
        assert bridge._copy_primary_text() is True

    copied_text = copy_mock.call_args[0][0]
    assert "line one" in copied_text
    assert "line two" in copied_text


def test_right_click_on_output_pastes_clipboard_into_input(tmp_path):
    cfg = _make_config(tmp_path)
    bridge = CLIBridge(SimpleNamespace(_config=cfg), options={})
    with patch("bridges.cli.__main__.Application", return_value=SimpleNamespace()):
        bridge._build_application()

    assert bridge._output_area is not None
    assert bridge._input_area is not None
    bridge._application = SimpleNamespace(
        layout=SimpleNamespace(focus=MagicMock()),
        invalidate=MagicMock(),
    )

    event = MouseEvent(
        position=Point(x=0, y=0),
        event_type=MouseEventType.MOUSE_UP,
        button=MouseButton.RIGHT,
        modifiers=frozenset(),
    )

    with patch.object(bridge, "_read_clipboard_text", return_value="hello"):
        assert bridge._output_area.control.mouse_handler(event) is None

    assert bridge._input_area.text == "[Pasted text #1, 5 chars]"
    bridge._application.layout.focus.assert_called_with(bridge._input_area)
    bridge._application.invalidate.assert_called()


def test_right_click_on_output_copies_selection_before_pasting(tmp_path):
    cfg = _make_config(tmp_path)
    bridge = CLIBridge(SimpleNamespace(_config=cfg), options={})
    with patch("bridges.cli.__main__.Application", return_value=SimpleNamespace()):
        bridge._build_application()

    assert bridge._output_area is not None
    bridge._output_area.buffer.set_document(
        Document(text="picked output", cursor_position=len("picked output")),
        bypass_readonly=True,
    )
    bridge._output_area.buffer.selection_state = SelectionState(
        original_cursor_position=0,
        type=SelectionType.CHARACTERS,
    )
    bridge._application = SimpleNamespace(
        layout=SimpleNamespace(focus=MagicMock()),
        invalidate=MagicMock(),
    )

    event = MouseEvent(
        position=Point(x=0, y=0),
        event_type=MouseEventType.MOUSE_UP,
        button=MouseButton.RIGHT,
        modifiers=frozenset(),
    )

    with patch.object(bridge, "_write_clipboard_text", return_value=True) as copy_mock:
        assert bridge._output_area.control.mouse_handler(event) is None

    copy_mock.assert_called_once_with("picked output")


def test_help_mentions_right_click_paste(tmp_path):
    cfg = _make_config(tmp_path)
    bridge = CLIBridge(SimpleNamespace(_config=cfg), options={})

    asyncio.run(bridge._handle_command("/help"))

    assert "Right click  copy selection, otherwise paste clipboard" in bridge._transcript_blocks[-1]
    assert "Ctrl+C       copy selected text or the transcript" in bridge._transcript_blocks[-1]
    assert "Ctrl+V       paste clipboard into input" in bridge._transcript_blocks[-1]
    assert "Ctrl+T       toggle mouse capture / selection mode" in bridge._transcript_blocks[-1]
    assert "PgUp/PgDn    scroll transcript" in bridge._transcript_blocks[-1]
    assert "Mouse capture off keeps terminal-native drag-select copy available" in bridge._transcript_blocks[-1]
    assert "/mouse       toggle mouse capture / selection mode" in bridge._transcript_blocks[-1]


def test_copy_command_copies_last_tool_block(tmp_path):
    cfg = _make_config(tmp_path)
    bridge = CLIBridge(SimpleNamespace(_config=cfg), options={})
    bridge._transcript_blocks = [
        "› hello",
        "[tool shell Get-ChildItem -Path C:\\repo]",
        "[err shell [stderr] missing path [exit 1]]",
    ]
    with patch.object(bridge, "_write_clipboard_text", return_value=True) as copy_mock:
        asyncio.run(bridge._handle_command("/copy last-tool"))

    copy_mock.assert_called_once_with(
        "[tool shell Get-ChildItem -Path C:\\repo]\n[err shell [stderr] missing path [exit 1]]"
    )
    assert bridge._transcript_blocks[-1] == "copied last tool block"


def test_copy_command_copies_last_error_block(tmp_path):
    cfg = _make_config(tmp_path)
    bridge = CLIBridge(SimpleNamespace(_config=cfg), options={})
    bridge._transcript_blocks = [
        "[tool shell bad command]",
        "[err shell [stderr] boom [exit 1]]",
        "› next",
    ]
    with patch.object(bridge, "_write_clipboard_text", return_value=True) as copy_mock:
        asyncio.run(bridge._handle_command("/copy last-error"))

    copy_mock.assert_called_once_with(
        "[tool shell bad command]\n[err shell [stderr] boom [exit 1]]"
    )
    assert bridge._transcript_blocks[-1] == "copied last error block"


def test_copy_command_errors_alias_copies_last_error_block(tmp_path):
    cfg = _make_config(tmp_path)
    bridge = CLIBridge(SimpleNamespace(_config=cfg), options={})
    bridge._transcript_blocks = [
        "[tool shell bad command]",
        "[err shell [stderr] boom [exit 1]]",
    ]
    with patch.object(bridge, "_write_clipboard_text", return_value=True) as copy_mock:
        asyncio.run(bridge._handle_command("/copy errors"))

    copy_mock.assert_called_once_with(
        "[tool shell bad command]\n[err shell [stderr] boom [exit 1]]"
    )


def test_copy_command_uses_raw_tool_history_when_available(tmp_path):
    cfg = _make_config(tmp_path)
    bridge = CLIBridge(SimpleNamespace(_config=cfg), options={})
    bridge._record_tool_call("call-1", "shell", {"command": 'Get-Content "C:\\repo\\README.md" -Head 50'})
    bridge._record_tool_result("call-1", "shell", "line one\nline two\nline three", False)

    with patch.object(bridge, "_write_clipboard_text", return_value=True) as copy_mock:
        asyncio.run(bridge._handle_command("/copy last-tool"))

    copy_mock.assert_called_once_with(
        'tool shell(command=\'Get-Content "C:\\\\repo\\\\README.md" -Head 50\')\n'
        'ok shell\nline one\nline two\nline three'
    )
    assert bridge._transcript_blocks[-1] == "copied last tool block"


def test_copy_command_can_copy_last_tool_call_and_result(tmp_path):
    cfg = _make_config(tmp_path)
    bridge = CLIBridge(SimpleNamespace(_config=cfg), options={})
    bridge._record_tool_call("call-1", "web_search", {"query": "NHL scores today"})
    bridge._record_tool_result("call-1", "web_search", "Search results...", False)

    with patch.object(bridge, "_write_clipboard_text", return_value=True) as copy_mock:
        asyncio.run(bridge._handle_command("/copy last-tool-call"))
        asyncio.run(bridge._handle_command("/copy last-tool-result"))

    assert copy_mock.call_args_list[0][0][0] == "tool web_search(query='NHL scores today')"
    assert copy_mock.call_args_list[1][0][0] == "ok web_search\nSearch results..."


def test_copy_command_can_copy_all_tool_blocks(tmp_path):
    cfg = _make_config(tmp_path)
    bridge = CLIBridge(SimpleNamespace(_config=cfg), options={})
    bridge._record_tool_call("call-1", "shell", {"command": "pwd"})
    bridge._record_tool_result("call-1", "shell", "C:\\repo", False)
    bridge._record_tool_call("call-2", "shell", {"command": "dir"})
    bridge._record_tool_result("call-2", "shell", "file.txt", False)

    with patch.object(bridge, "_write_clipboard_text", return_value=True) as copy_mock:
        asyncio.run(bridge._handle_command("/copy all-tools"))

    copied_text = copy_mock.call_args[0][0]
    assert "tool shell(command='pwd')" in copied_text
    assert "tool shell(command='dir')" in copied_text
    assert copied_text.count("ok shell") == 2


def test_agent_error_resets_status_to_ready(tmp_path):
    cfg = _make_config(tmp_path)
    bridge = CLIBridge(SimpleNamespace(_config=cfg), options={})

    event = AgentError(
        tail_node_id="tail-1",
        lane_node_id="lane-1",
        trace_id="trace-1",
        reply_to_message_id="msg-1",
        message="[internal error]",
    )

    asyncio.run(bridge.handle_event(event))

    assert bridge._transcript_blocks[-1] == "error: [internal error]"
    assert bridge._footer_text() == "working ready | mouse"


def test_tool_result_keeps_status_as_thinking(tmp_path):
    cfg = _make_config(tmp_path)
    bridge = CLIBridge(SimpleNamespace(_config=cfg), options={})

    event = AgentToolResult(
        tail_node_id="tail-1",
        lane_node_id="lane-1",
        trace_id="trace-1",
        reply_to_message_id="msg-1",
        call_id="call-1",
        tool_name="shell",
        output="C:\\repo",
        is_error=False,
    )

    asyncio.run(bridge.handle_event(event))

    assert bridge._transcript_blocks[-1] == "[ok shell C:\\repo]"
    assert bridge._footer_text() == "working thinking | mouse"


def test_debug_alias_routes_to_heartbeat(tmp_path):
    cfg = _make_config(tmp_path)
    bridge = CLIBridge(SimpleNamespace(_config=cfg), options={})

    with patch("bridges.cli.__main__._debug_heartbeat", new=AsyncMock()) as debug_mock:
        asyncio.run(bridge._handle_command("/debug"))

    assert debug_mock.await_count == 1


def test_cli_restores_transcript_from_saved_cursor(tmp_path):
    cfg = _make_config(tmp_path)
    gateway = SimpleNamespace(_config=cfg)
    bridge = CLIBridge(gateway, options={})

    db = ConversationDB(tmp_path / "agent.db")
    root = db.get_root()
    session = db.add_node(parent_id=root.id, role="system", content="session:cli")
    user = db.add_node(parent_id=session.id, role="user", content="hello")
    assistant = db.add_node(parent_id=user.id, role="assistant", content="hi there")
    bridge._cursor = assistant.id
    db.close()

    restored = bridge._restore_transcript_from_cursor()

    assert restored == 2
    assert bridge._transcript_blocks == ["› hello", "hi there"]


def test_cli_restores_tool_history_from_saved_cursor(tmp_path):
    cfg = _make_config(tmp_path)
    gateway = SimpleNamespace(_config=cfg)
    bridge = CLIBridge(gateway, options={})

    db = ConversationDB(tmp_path / "agent.db")
    root = db.get_root()
    session = db.add_node(parent_id=root.id, role="system", content="session:cli")
    user = db.add_node(parent_id=session.id, role="user", content="check cwd")
    assistant = db.add_node(
        parent_id=user.id,
        role="assistant",
        content="",
        tool_calls=json.dumps([
            {"id": "call-1", "name": "shell", "arguments": {"command": "pwd"}}
        ]),
    )
    tool = db.add_node(
        parent_id=assistant.id,
        role="tool",
        content="C:\\repo",
        tool_call_id="call-1",
    )
    bridge._cursor = tool.id
    db.close()

    restored = bridge._restore_transcript_from_cursor()

    assert restored == 3
    assert bridge._transcript_blocks[0] == "› check cwd"
    assert bridge._transcript_blocks[1] == "[tool shell pwd]"
    assert bridge._transcript_blocks[2] == "[ok shell C:\\repo]"


def test_cli_restores_shell_error_history_as_err(tmp_path):
    cfg = _make_config(tmp_path)
    gateway = SimpleNamespace(_config=cfg)
    bridge = CLIBridge(gateway, options={})

    db = ConversationDB(tmp_path / "agent.db")
    root = db.get_root()
    session = db.add_node(parent_id=root.id, role="system", content="session:cli")
    user = db.add_node(parent_id=session.id, role="user", content="check missing path")
    assistant = db.add_node(
        parent_id=user.id,
        role="assistant",
        content="",
        tool_calls=json.dumps([
            {"id": "call-1", "name": "shell", "arguments": {"command": "Get-ChildItem -Path C:\\missing"}}
        ]),
    )
    tool = db.add_node(
        parent_id=assistant.id,
        role="tool",
        content="[stderr]\nGet-ChildItem : Cannot find path\n[exit 1]",
        tool_call_id="call-1",
    )
    bridge._cursor = tool.id
    db.close()

    restored = bridge._restore_transcript_from_cursor()

    assert restored == 3
    assert bridge._transcript_blocks[2].startswith("[err shell")


def test_cli_restore_non_shell_exit_marker_stays_ok(tmp_path):
    cfg = _make_config(tmp_path)
    gateway = SimpleNamespace(_config=cfg)
    bridge = CLIBridge(gateway, options={})

    db = ConversationDB(tmp_path / "agent.db")
    root = db.get_root()
    session = db.add_node(parent_id=root.id, role="system", content="session:cli")
    user = db.add_node(parent_id=session.id, role="user", content="show logs")
    assistant = db.add_node(
        parent_id=user.id,
        role="assistant",
        content="",
        tool_calls=json.dumps([
            {"id": "call-1", "name": "my_tool", "arguments": {}}
        ]),
    )
    tool = db.add_node(
        parent_id=assistant.id,
        role="tool",
        content="build log line\n[exit 1]",
        tool_call_id="call-1",
    )
    bridge._cursor = tool.id
    db.close()

    restored = bridge._restore_transcript_from_cursor()

    assert restored == 3
    assert bridge._transcript_blocks[2].startswith("[ok my_tool")
