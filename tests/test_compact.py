from context import HistoryEntry
from contracts import ToolCall, ToolResult
from compact import build_compaction_plan, format_summary


class TestCompactionPlanning:
    def test_tool_call_group_is_not_split_from_tool_result_tail(self):
        tc = ToolCall(call_id="c1", tool_name="search", args={"q": "cats"})
        tool_result = ToolResult(call_id="c1", tool_name="search", output="cats found")
        entries = [
            HistoryEntry.user("older user"),
            HistoryEntry.assistant("older reply"),
            HistoryEntry.user("recent request"),
            HistoryEntry.assistant("calling", tool_calls=[tc]),
            HistoryEntry.tool_result(tool_result),
        ]

        plan = build_compaction_plan(entries, keep_last_units=1)

        assert plan is not None
        assert [e.role for e in plan.to_summarize] == ["user", "assistant", "user"]
        assert [e.role for e in plan.preserved_tail] == ["assistant", "tool"]
        assert plan.preserved_tail[0].tool_calls[0]["id"] == "c1"
        assert plan.preserved_tail[1].tool_call_id == "c1"

    def test_format_summary_adds_header_and_truncates(self):
        formatted = format_summary("x" * 7000)
        assert formatted.startswith("Compact summary:\n")
        assert formatted.endswith("...")
