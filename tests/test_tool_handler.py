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
# get_tool_definitions()
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

        self.handler.register_tool(search)
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

    def test_multiple_tools_all_returned(self):
        def a() -> str:
            """A."""
            return ""

        def b() -> str:
            """B."""
            return ""

        self.handler.register_tool(a)
        self.handler.register_tool(b)
        defs = self.handler.get_tool_definitions()
        names = {d["function"]["name"] for d in defs}
        assert names == {"a", "b"}


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

        self.handler.register_tool(add)
        result = await self.handler.execute_tool_call({
            "id": "call1",
            "function": {"name": "add", "arguments": '{"a": 3, "b": 4}'}
        })
        assert result["success"] is True
        assert result["result"] == "7"

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