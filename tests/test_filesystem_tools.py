"""
tests/test_filesystem_tools.py

Tests for the filesystem module's new tools and guards:
  - replace_all on str_replace
  - grep (ripgrep wrapper / Python fallback)
  - glob_search
  - read-before-write guard
  - file staleness detection

Run with:
    python -m pytest tests/test_filesystem_tools.py -v
"""
from __future__ import annotations

import os
import time
import tempfile
import shutil
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers — register the filesystem module against a temp workspace
# ---------------------------------------------------------------------------

class _MockConfig:
    def __init__(self, ws_path: str):
        self.workspace = type("WS", (), {"path": ws_path})()
        self.shell_timeout = 60


class _MockToolHandler:
    def __init__(self):
        self.tools: dict = {}

    def register_tool(self, func, always_on=False):
        self.tools[func.__name__] = func


class _MockAgent:
    def __init__(self, ws_path: str):
        self.config = _MockConfig(ws_path)
        self.tool_handler = _MockToolHandler()


@pytest.fixture
def workspace(tmp_path):
    """Create a temp workspace with some test files."""
    # Create files for testing
    (tmp_path / "hello.py").write_text("def hello():\n    return 'world'\n", encoding="utf-8")
    (tmp_path / "greet.py").write_text("def greet(name):\n    return f'hi {name}'\n", encoding="utf-8")
    (tmp_path / "readme.md").write_text("# Project\nSome readme text.\n", encoding="utf-8")

    sub = tmp_path / "src"
    sub.mkdir()
    (sub / "main.py").write_text("import hello\nhello.hello()\n", encoding="utf-8")
    (sub / "utils.ts").write_text("export function add(a: number, b: number) { return a + b; }\n", encoding="utf-8")

    # .git dir — should be excluded from search
    git = tmp_path / ".git"
    git.mkdir()
    (git / "config").write_text("[core]\nrepositoryformatversion = 0\n", encoding="utf-8")

    return tmp_path


@pytest.fixture
def tools(workspace):
    """Register the filesystem module and return the tools dict."""
    import sys
    # Ensure project root is on path so 'contracts' and 'modules' are importable
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from modules.filesystem.__main__ import register

    agent = _MockAgent(str(workspace))
    register(agent)
    return agent.tool_handler.tools


# ===================================================================
# str_replace — replace_all flag
# ===================================================================

class TestStrReplaceAll:
    def test_single_match_works(self, tools, workspace):
        tools["view"](path="hello.py")
        result = tools["str_replace"](path="hello.py", old_str="world", new_str="earth")
        assert "replaced 1 occurrence" in result
        content = (workspace / "hello.py").read_text(encoding="utf-8")
        assert "earth" in content

    def test_multiple_matches_blocked_by_default(self, tools, workspace):
        # Write a file with repeated content
        (workspace / "dup.txt").write_text("aaa bbb aaa ccc aaa", encoding="utf-8")
        tools["view"](path="dup.txt")
        result = tools["str_replace"](path="dup.txt", old_str="aaa")
        assert "appears 3 times" in result
        assert "replace_all" in result

    def test_replace_all_replaces_every_occurrence(self, tools, workspace):
        (workspace / "dup.txt").write_text("aaa bbb aaa ccc aaa", encoding="utf-8")
        tools["view"](path="dup.txt")
        result = tools["str_replace"](path="dup.txt", old_str="aaa", new_str="zzz", replace_all=True)
        assert "replaced 3 occurrences" in result
        content = (workspace / "dup.txt").read_text(encoding="utf-8")
        assert content == "zzz bbb zzz ccc zzz"

    def test_replace_all_with_single_match(self, tools, workspace):
        tools["view"](path="hello.py")
        result = tools["str_replace"](path="hello.py", old_str="world", new_str="earth", replace_all=True)
        assert "replaced 1 occurrence" in result

    def test_replace_all_not_found(self, tools, workspace):
        tools["view"](path="hello.py")
        result = tools["str_replace"](path="hello.py", old_str="zzznomatch")
        assert "not found" in result


# ===================================================================
# grep — search functionality
# ===================================================================

class TestGrep:
    def test_files_mode_default(self, tools):
        result = tools["grep"](pattern="def ")
        assert "hello.py" in result
        assert "greet.py" in result

    def test_files_mode_excludes_git(self, tools):
        result = tools["grep"](pattern="core")
        # .git/config contains "core" but should be excluded
        assert ".git" not in result

    def test_content_mode(self, tools):
        result = tools["grep"](pattern="def hello", output_mode="content")
        assert "def hello" in result
        assert "hello.py" in result

    def test_count_mode(self, tools):
        result = tools["grep"](pattern="import", output_mode="count")
        assert "matches" in result.lower() or "1" in result

    def test_no_matches(self, tools):
        result = tools["grep"](pattern="zzznomatchzzz")
        assert "no matches" in result.lower()

    def test_include_filter(self, tools):
        result = tools["grep"](pattern="def", include="*.py")
        assert "hello.py" in result
        # .ts file should not appear
        assert "utils.ts" not in result

    def test_case_insensitive(self, tools):
        result = tools["grep"](pattern="PROJECT", case_insensitive=True)
        assert "readme.md" in result

    def test_context_lines(self, tools):
        # Search a directory so rg doesn't need special single-file handling
        result = tools["grep"](pattern="return", include="hello.py", output_mode="content", context_lines=1)
        # Should include the match line at minimum
        assert "return" in result

    def test_bad_path(self, tools):
        result = tools["grep"](pattern="test", path="nonexistent_dir")
        assert "error" in result.lower()

    def test_limit_caps_results(self, tools):
        result = tools["grep"](pattern=".", output_mode="content", limit=3)
        lines = [l for l in result.splitlines() if l.strip()]
        # Should be capped (may include truncation notice)
        assert len(lines) <= 5  # 3 + possible header/footer

    def test_subdirectory_search(self, tools):
        result = tools["grep"](pattern="import", path="src")
        assert "main.py" in result


# ===================================================================
# glob_search — file finding
# ===================================================================

class TestGlobSearch:
    def test_find_python_files(self, tools):
        result = tools["glob_search"](pattern="**/*.py")
        assert "hello.py" in result
        assert "greet.py" in result
        assert "main.py" in result

    def test_find_markdown_files(self, tools):
        result = tools["glob_search"](pattern="*.md")
        assert "readme.md" in result

    def test_excludes_git_dir(self, tools):
        result = tools["glob_search"](pattern="**/*")
        assert ".git" not in result

    def test_no_files_found(self, tools):
        result = tools["glob_search"](pattern="*.zzz")
        assert "no files found" in result.lower()

    def test_bad_path(self, tools):
        result = tools["glob_search"](pattern="*.py", path="nonexistent")
        assert "error" in result.lower()

    def test_subdirectory_search(self, tools):
        result = tools["glob_search"](pattern="*.py", path="src")
        assert "main.py" in result
        # Should not include files from parent
        assert "hello.py" not in result

    def test_limit(self, tools):
        result = tools["glob_search"](pattern="**/*", limit=2)
        lines = result.strip().splitlines()
        # Header line + at most 2 file lines
        file_lines = [l for l in lines if not l.startswith("[")]
        assert len(file_lines) <= 2

    def test_sorted_by_mtime(self, tools, workspace):
        # Touch hello.py to make it newest
        time.sleep(0.05)
        (workspace / "hello.py").write_text("# touched\n", encoding="utf-8")
        result = tools["glob_search"](pattern="*.py")
        lines = [l for l in result.strip().splitlines() if not l.startswith("[")]
        # hello.py should be first (newest)
        assert lines[0] == "hello.py"


# ===================================================================
# Read-before-write guard
# ===================================================================

class TestReadBeforeWrite:
    def test_str_replace_blocked_without_read(self, tools, workspace):
        result = tools["str_replace"](path="hello.py", old_str="world", new_str="earth")
        assert "not been read" in result
        # File should be unchanged
        assert "world" in (workspace / "hello.py").read_text(encoding="utf-8")

    def test_str_replace_allowed_after_read(self, tools, workspace):
        tools["view"](path="hello.py")
        result = tools["str_replace"](path="hello.py", old_str="world", new_str="earth")
        assert "replaced" in result

    def test_write_file_blocked_for_existing_without_read(self, tools, workspace):
        result = tools["write_file"](path="hello.py", content="overwritten!", mode="overwrite")
        assert "not been read" in result
        # File should be unchanged
        assert "def hello" in (workspace / "hello.py").read_text(encoding="utf-8")

    def test_write_file_allowed_for_new_file(self, tools, workspace):
        result = tools["write_file"](path="brand_new.txt", content="new content", mode="overwrite")
        assert "created" in result
        assert (workspace / "brand_new.txt").read_text(encoding="utf-8") == "new content"

    def test_write_after_write_works(self, tools, workspace):
        """After write_file creates a file, a second write should work (mtime tracked)."""
        tools["write_file"](path="new.txt", content="first", mode="overwrite")
        result = tools["write_file"](path="new.txt", content="second", mode="overwrite")
        assert "overwrote" in result
        assert (workspace / "new.txt").read_text(encoding="utf-8") == "second"

    def test_str_replace_after_str_replace_works(self, tools, workspace):
        """Back-to-back edits should work (mtime updated after each write)."""
        tools["view"](path="hello.py")
        tools["str_replace"](path="hello.py", old_str="world", new_str="earth")
        result = tools["str_replace"](path="hello.py", old_str="earth", new_str="mars")
        assert "replaced" in result


# ===================================================================
# File staleness detection
# ===================================================================

class TestStalenessDetection:
    def test_external_modification_blocks_write(self, tools, workspace):
        tools["view"](path="hello.py")
        # Simulate external modification
        time.sleep(0.05)
        (workspace / "hello.py").write_text("# externally changed\n", encoding="utf-8")
        result = tools["str_replace"](path="hello.py", old_str="externally", new_str="changed")
        assert "modified since" in result

    def test_re_read_clears_staleness(self, tools, workspace):
        tools["view"](path="hello.py")
        time.sleep(0.05)
        (workspace / "hello.py").write_text("# externally changed\n", encoding="utf-8")
        # First attempt blocked
        result = tools["str_replace"](path="hello.py", old_str="externally", new_str="internally")
        assert "modified since" in result
        # Re-read
        tools["view"](path="hello.py")
        # Second attempt succeeds
        result = tools["str_replace"](path="hello.py", old_str="externally", new_str="internally")
        assert "replaced" in result

    def test_write_file_staleness(self, tools, workspace):
        tools["view"](path="readme.md")
        time.sleep(0.05)
        (workspace / "readme.md").write_text("# changed externally\n", encoding="utf-8")
        result = tools["write_file"](path="readme.md", content="overwrite", mode="overwrite")
        assert "modified since" in result
