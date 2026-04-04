"""
compact.py — Lightweight conversation compaction planning for TinyCTX.

This is intentionally simple:
  - keep a small recent raw suffix
  - summarize the older prefix
  - never split assistant tool_calls from their tool_result tail
"""

from __future__ import annotations

from dataclasses import dataclass

from context import HistoryEntry, ROLE_ASSISTANT, ROLE_TOOL

COMPACT_TRIGGER_PCT = 0.90
COMPACT_KEEP_LAST_UNITS = 4
COMPACT_SUMMARY_MAX_CHARS = 6000

COMPACT_SYSTEM_PROMPT = (
    "You are compressing older TinyCTX conversation history so work can continue "
    "without losing important context. Summarize concrete user goals, constraints, "
    "code/files touched, tool findings, errors, decisions, and pending work. "
    "Keep it dense and factual. Output plain text only."
)

COMPACT_USER_PROMPT = (
    "Summarize the conversation above for future continuation. The most recent raw "
    "turns will remain outside this summary, so focus on durable older context."
)


@dataclass
class CompactionPlan:
    to_summarize: list[HistoryEntry]
    preserved_tail: list[HistoryEntry]
    summarized_units: int
    preserved_units: int


def should_compact(tokens_used: int, token_limit: int) -> bool:
    if token_limit <= 0:
        return False
    return (tokens_used / token_limit) >= COMPACT_TRIGGER_PCT


def build_compaction_plan(
    entries: list[HistoryEntry],
    *,
    keep_last_units: int = COMPACT_KEEP_LAST_UNITS,
) -> CompactionPlan | None:
    units = _group_entries(entries)
    if len(units) <= keep_last_units:
        return None

    summarized_units = units[:-keep_last_units]
    preserved_units = units[-keep_last_units:]
    to_summarize = [entry for unit in summarized_units for entry in unit]
    preserved_tail = [entry for unit in preserved_units for entry in unit]

    if not to_summarize:
        return None

    return CompactionPlan(
        to_summarize=to_summarize,
        preserved_tail=preserved_tail,
        summarized_units=len(summarized_units),
        preserved_units=len(preserved_units),
    )


def format_summary(summary: str) -> str:
    summary = (summary or "").strip()
    if not summary:
        summary = "- Earlier context was compacted, but the summary came back empty."
    if len(summary) > COMPACT_SUMMARY_MAX_CHARS:
        summary = summary[: COMPACT_SUMMARY_MAX_CHARS - 3].rstrip() + "..."
    return "Compact summary:\n" + summary


def _group_entries(entries: list[HistoryEntry]) -> list[list[HistoryEntry]]:
    groups: list[list[HistoryEntry]] = []
    i = 0

    while i < len(entries):
        entry = entries[i]
        if entry.role == ROLE_ASSISTANT and entry.tool_calls:
            group = [entry]
            call_ids = {tc["id"] for tc in entry.tool_calls}
            i += 1
            while i < len(entries):
                candidate = entries[i]
                if candidate.role != ROLE_TOOL or candidate.tool_call_id not in call_ids:
                    break
                group.append(candidate)
                i += 1
            groups.append(group)
            continue

        groups.append([entry])
        i += 1

    return groups
