"""
tests/test_tool_handler.py

Tests for ToolCallHandler — registration, schema extraction, and execution.

Run with:
    pytest tests/
"""
import pytest
from utils.tool_handler import ToolCallHandler


# ---------------------------------------------------------------------------
# Registration and schema extraction
# ---------------------------------------------------------------------------

class TestToolRegistration:
    def setup_method(self):
        self.handler = ToolCallHandler()

    def test_register_simple_function(self):
        def greet(name: str) -> str:
            """Say hello to someone.

            Args:
                name: The person's name.
            """
            return f"Hello, {name}!"

        self.handler.register_tool(greet)
        assert "greet" in self.handler.tools

    def test_description_extracted_from_docstring(self):
        def greet(name: str) -> str:
            """Say hello to someone.

            Args:
                name: The person's name.
            """
            return f"Hello, {name}!"

        self.handler.register_tool(greet)
        assert self.handler.tools["greet"]["description"] == "Say hello to someone."

    def test_arg_description_extracted(self):
        def greet(name: str) -> str:
            """Say hello.

            Args:
                name: The person's name.
            """
            return f"Hello, {name}!"

        self.handler.register_tool(greet)
        assert "description" in self.handler.tools["greet"]["properties"]["name"]
        assert "name" in self.handler.tools["greet"]["properties"]["name"]["description"].lower()

    def test_required_args_captured(self):
        def fn(required_arg: str, optional_arg: str = "default") -> str:
            """A function."""
            return required_arg

        self.handler.register_tool(fn)
        tool = self.handler.tools["fn"]
        assert "required_arg" in tool["required"]
        assert "optional_arg" not in tool["required"]

    def test_type_annotations_mapped(self):
        def fn(s: str, i: int, f: float, b: bool, d: dict, lst: list) -> str:
            """Types test."""
            return ""

        self.handler.register_tool(fn)
        props = self.handler.tools["fn"]["properties"]
        assert props["s"]["type"] == "string"
        assert props["i"]["type"] == "integer"
        assert props["f"]["type"] == "number"
        assert props["b"]["type"] == "boolean"
        assert props["d"]["type"] == "object"
        assert props["lst"]["type"] == "array"

    def test_no_docstring_falls_back_gracefully(self):
        def nodoc(x: str) -> str:
            return x

        self.handler.register_tool(nodoc)
        assert "nodoc" in self.handler.tools
        assert self.handler.tools["nodoc"]["description"]  # not empty

    def test_custom_name_override(self):
        def fn() -> str:
            """Does something."""
            return ""

        self.handler.register_tool(fn, name="custom_name")
        assert "custom_name" in self.handler.tools
        assert "fn" not in self.handler.tools

    def test_custom_description_override(self):
        def fn() -> str:
            """Original docstring."""
            return ""

        self.handler.register_tool(fn, description="My custom description")
        assert self.handler.tools["fn"]["description"] == "My custom description"


# ---------------------------------------------------------------------------
# always_on / deferred registration
# ---------------------------------------------------------------------------

class TestAlwaysOnDeferred:
    def setup_method(self):
        self.handler = ToolCallHandler()

    def test_always_on_immediately_enabled(self):
        def fn() -> str:
            """Always on."""
            return ""
        self.handler.register_tool(fn, always_on=True)
        assert "fn" in self.handler.enabled

    def test_deferred_not_in_enabled(self):
        def fn() -> str:
            """Deferred."""
            return ""
        self.handler.register_tool(fn)  # always_on defaults to False
        assert "fn" not in self.handler.enabled

    def test_deferred_still_in_tools(self):
        """Deferred tools are registered in self.tools even though not enabled."""
        def fn() -> str:
            """Deferred."""
            return ""
        self.handler.register_tool(fn)
        assert "fn" in self.handler.tools

    def test_enable_method(self):
        def fn() -> str:
            """Tool."""
            return ""
        self.handler.register_tool(fn)
        assert "fn" not in self.handler.enabled
        result = self.handler.enable("fn")
        assert result is True
        assert "fn" in self.handler.enabled

    def test_enable_unknown_returns_false(self):
        assert self.handler.enable("nonexistent") is False


# ---------------------------------------------------------------------------
# tools_search()
# ---------------------------------------------------------------------------

class TestToolsSearch:
    def setup_method(self):
        self.handler = ToolCallHandler()
        # Register tools_search itself as always_on (mirrors agent.py bootstrap)
        self.handler.register_tool(self.handler.tools_search, always_on=True)

    def _add(self, name: str, description: str, always_on: bool = False):
        """Helper: register an anonymous tool with given name and description."""
        def fn() -> str:
            return ""
        fn.__name__ = name
        fn.__doc__ = description
        self.handler.register_tool(fn, always_on=always_on)

    def test_search_enables_matching_tool(self):
        self._add("web_search", "Search the web for information")
        result = self.handler.tools_search("web")
        assert "web_search" in self.handler.enabled
        assert "web_search" in result

    def test_search_matches_description(self):
        self._add("fetch_page", "Download and return the HTML of a URL")
        self.handler.tools_search("HTML")
        assert "fetch_page" in self.handler.enabled

    def test_search_case_insensitive(self):
        self._add("screenshot", "Take a screenshot of the page")
        self.handler.tools_search("SCREENSHOT")
        assert "screenshot" in self.handler.enabled

    def test_search_no_match_returns_message(self):
        result = self.handler.tools_search("zzznomatch")
        assert "No" in result or "no" in result

    def test_search_skips_already_enabled(self):
        self._add("already", "Already enabled tool", always_on=True)
        result = self.handler.tools_search("already")
        # Should not re-add, should report it's already enabled
        assert "Already" in result or "already" in result.lower()

    def test_search_multiple_matches(self):
        self._add("click", "Click an element on the page")
        self._add("double_click", "Double-click an element")
        self.handler.tools_search("click")
        assert "click" in self.handler.enabled
        assert "double_click" in self.handler.enabled

    def test_tools_search_itself_always_on(self):
        """tools_search should always be in the tool definitions."""
        defs = self.handler.get_tool_definitions()
        names = {d["function"]["name"] for d in defs}
        assert "tools_search" in names


# ---------------------------------------------------------------------------
# get_tool_definitions() — now filters to enabled set
# ---------------------------------------------------------------------------

class TestToolDefinitions:
    def setup_method(self):
        self.handler = ToolCallHandler()

    def test_definitions_format(self):
        def search(query: str) -> str:
            """Search the web.

            Args:
                query: What to search for.
            """
            return ""

        self.handler.register_tool(search, always_on=True)
        defs = self.handler.get_tool_definitions()

        assert len(defs) == 1
        d = defs[0]
        assert d["type"] == "function"
        assert d["function"]["name"] == "search"
        assert "description" in d["function"]
        assert d["function"]["parameters"]["type"] == "object"
        assert "query" in d["function"]["parameters"]["properties"]

    def test_empty_handler_returns_empty_list(self):
        assert self.handler.get_tool_definitions() == []

    def test_deferred_tools_not_in_definitions(self):
        """Deferred (not always_on) tools must not appear in definitions."""
        def secret() -> str:
            """A deferred tool."""
            return ""
        self.handler.register_tool(secret)  # deferred
        assert self.handler.get_tool_definitions() == []

    def test_only_enabled_tools_returned(self):
        def a() -> str:
            """A."""
            return ""
        def b() -> str:
            """B."""
            return ""
        def c() -> str:
            """C — deferred."""
            return ""

        self.handler.register_tool(a, always_on=True)
        self.handler.register_tool(b, always_on=True)
        self.handler.register_tool(c)  # deferred
        defs = self.handler.get_tool_definitions()
        names = {d["function"]["name"] for d in defs}
        assert names == {"a", "b"}
        assert "c" not in names

    def test_enable_then_appears_in_definitions(self):
        """Enabling a deferred tool mid-session makes it appear in next definitions call."""
        def lazy() -> str:
            """Lazy tool."""
            return ""
        self.handler.register_tool(lazy)
        assert self.handler.get_tool_definitions() == []
        self.handler.enable("lazy")
        defs = self.handler.get_tool_definitions()
        assert any(d["function"]["name"] == "lazy" for d in defs)


# ---------------------------------------------------------------------------
# execute_tool_call() — sync functions
# ---------------------------------------------------------------------------

class TestExecuteToolSync:
    def setup_method(self):
        self.handler = ToolCallHandler()

    @pytest.mark.asyncio
    async def test_execute_returns_result(self):
        def add(a: int, b: int) -> str:
            """Add two numbers."""
            return str(a + b)

        # execute_tool_call dispatches via self.tools (not self.enabled),
        # so deferred tools can still be called once the agent loop invokes them.
        self.handler.register_tool(add)
        result = await self.handler.execute_tool_call({
            "id": "call1",
            "function": {"name": "add", "arguments": '{"a": 3, "b": 4}'}
        })
        assert result["success"] is True
        assert result["result"] == "7"

    @pytest.mark.asyncio
    async def test_execute_deferred_tool_still_callable(self):
        """A deferred (not enabled) tool must still execute when called."""
        def multiply(a: int, b: int) -> str:
            """Multiply."""
            return str(a * b)

        self.handler.register_tool(multiply)  # deferred — not in enabled
        assert "multiply" not in self.handler.enabled
        result = await self.handler.execute_tool_call({
            "id": "c1",
            "function": {"name": "multiply", "arguments": '{"a": 6, "b": 7}'}
        })
        assert result["success"] is True
        assert result["result"] == "42"

    @pytest.mark.asyncio
    async def test_execute_unknown_tool_returns_error(self):
        result = await self.handler.execute_tool_call({
            "id": "call1",
            "function": {"name": "nonexistent", "arguments": "{}"}
        })
        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_invalid_json_returns_error(self):
        def fn(x: str) -> str:
            """A function."""
            return x

        self.handler.register_tool(fn)
        result = await self.handler.execute_tool_call({
            "id": "call1",
            "function": {"name": "fn", "arguments": "not valid json {{{"}
        })
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_execute_dict_args(self):
        """Arguments can be passed as a dict instead of a JSON string."""
        def greet(name: str) -> str:
            """Greet."""
            return f"hi {name}"

        self.handler.register_tool(greet)
        result = await self.handler.execute_tool_call({
            "id": "c1",
            "function": {"name": "greet", "arguments": {"name": "world"}}
        })
        assert result["success"] is True
        assert result["result"] == "hi world"

    @pytest.mark.asyncio
    async def test_execute_raises_captured_as_error(self):
        def boom(x: str) -> str:
            """Explodes."""
            raise ValueError("intentional failure")

        self.handler.register_tool(boom)
        result = await self.handler.execute_tool_call({
            "id": "c1",
            "function": {"name": "boom", "arguments": '{"x": "test"}'}
        })
        assert result["success"] is False
        assert "intentional failure" in result["error"]


# ---------------------------------------------------------------------------
# execute_tool_call() — async functions
# ---------------------------------------------------------------------------

class TestExecuteToolAsync:
    def setup_method(self):
        self.handler = ToolCallHandler()

    @pytest.mark.asyncio
    async def test_async_tool_awaited(self):
        import asyncio

        async def slow_add(a: int, b: int) -> str:
            """Async add."""
            await asyncio.sleep(0)
            return str(a + b)

        self.handler.register_tool(slow_add)
        result = await self.handler.execute_tool_call({
            "id": "c1",
            "function": {"name": "slow_add", "arguments": '{"a": 10, "b": 5}'}
        })
        assert result["success"] is True
        assert result["result"] == "15"

    @pytest.mark.asyncio
    async def test_async_tool_exception_captured(self):
        async def async_boom(x: str) -> str:
            """Async explodes."""
            raise RuntimeError("async failure")

        self.handler.register_tool(async_boom)
        result = await self.handler.execute_tool_call({
            "id": "c1",
            "function": {"name": "async_boom", "arguments": '{"x": "hi"}'}
        })
        assert result["success"] is False
        assert "async failure" in result["error"]