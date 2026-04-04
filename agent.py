"""
agent.py — The 6-stage agent execution loop.
One instance per session, owned by its Lane.
Yields AgentEvent objects; never calls the gateway directly.

Stages:
  1. Intake           — add user message to Context  (skipped when msg is None)
  2. Context Assembly — await async hooks, then build message list via Context.assemble()
  3. Inference        — stream LLM, collect text + tool calls
  4. Tool Execution   — dispatch ToolCalls via tool_handler
  5. Result Backfill  — inject ToolResults back into Context
  6. Streaming Reply  — yield AgentEvent objects to Lane

Streaming behaviour:
  Tool-call cycles buffer text. The final cycle streams AgentTextChunk events
  live with a closing AgentTextFinal. Tool-use cycles yield AgentToolCall /
  AgentToolResult so bridges can display tool activity live.

Abort:
  Lane passes its abort_event to run(). The loop checks it between every
  inference cycle and inside the LLM stream. If set, yields AgentError and
  exits cleanly.

Tree refactor (Phase 2):
  AgentLoop is initialised with tail_node_id (str cursor) instead of
  session_key. The cursor points at the DB branch this agent is running on.
  All context.add() calls write immediately to the DB; cursor is kept in sync
  on disk after every node write.

Background branches (Phase 3):
  Modules register post-turn hooks via register_background_hook(fn). After
  AgentTextFinal is yielded, each hook receives the current tail_node_id and
  the agent's config. Hooks run as detached asyncio tasks — they must not
  block the caller. The canonical use is memory consolidation: create an
  opening node off the current tail via db.add_node(), construct a new
  AgentLoop pointing at it, and run it to completion.

  _run_background(tail_node_id, config) — module-level helper that fires a
  standalone AgentLoop in a detached task and discards all events.

Model pool:
  All named chat models from config.models are pre-instantiated.
  Inference walks primary → fallback list per config.llm.fallback_on.
  Modules call agent.get_model(name) to get a named LLM instance.

Module loading:
  Scans modules/ for packages exposing register(agent). No hardcoding.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import re
import uuid
from pathlib import Path
from typing import AsyncIterator, Callable, Awaitable

from contracts import (
    InboundMessage, AgentEvent,
    AgentThinkingChunk, AgentTextChunk, AgentTextFinal, AgentToolCall, AgentToolResult, AgentError,
    ToolCall, ToolResult, IMAGE_BLOCK_PREFIX,
)
from context import Context, HistoryEntry, HOOK_PRE_ASSEMBLE_ASYNC
from config import Config, ModelConfig
from ai import LLM, TextDelta, ThinkingDelta, ToolCallAssembled, LLMError
from utils.tool_handler import ToolCallHandler
from utils.attachments import build_content_blocks
from db import ConversationDB
from compact import (
    COMPACT_SYSTEM_PROMPT,
    COMPACT_USER_PROMPT,
    build_compaction_plan,
    format_summary,
    should_compact,
)

logger = logging.getLogger(__name__)

MODULES_DIR = Path("modules")
_SHELL_EXIT_ERROR_RE_END = re.compile(r"(?:^|\n)\[exit \d+\]\s*\Z")
_EMPTY_REPLY_RETRY_PROMPT = (
    "You already have the tool results for this turn. "
    "If you do not need another tool, answer the user directly now. "
    "Do not return an empty response."
)


def _tool_cache_key(call: ToolCall) -> str:
    return call.tool_name + "::" + json.dumps(call.args, sort_keys=True, ensure_ascii=False)


def _summarize_cached_tool_call(call: ToolCall, *, max_chars: int = 160) -> str:
    if not call.args:
        return f"{call.tool_name}()"
    parts = [
        f"{key}={json.dumps(value, ensure_ascii=False)}"
        for key, value in sorted(call.args.items())
    ]
    summary = f"{call.tool_name}(" + ", ".join(parts) + ")"
    summary = summary.replace("\n", " ")
    if len(summary) > max_chars:
        return summary[: max_chars - 1] + "…"
    return summary


def _cached_tool_result_notice(call: ToolCall, *, is_error: bool) -> str:
    if is_error:
        return (
            "[error: cached exact same tool call reused earlier error from this turn: "
            + _summarize_cached_tool_call(call)
            + " — refer to the previous tool result instead of calling it again]"
        )
    return (
        "[cached exact same tool call reused earlier result from this turn: "
        + _summarize_cached_tool_call(call)
        + " — refer to the previous tool result instead of calling it again]"
    )


def _looks_like_failed_shell_output(output: str) -> bool:
    lowered = (output or "").lstrip().lower()
    if lowered.startswith("[error") or lowered.startswith("[blocked"):
        return True
    return bool(_SHELL_EXIT_ERROR_RE_END.search(output or ""))


def _normalize_error_output(tool_name: str, output: str) -> str:
    text = (output or "").strip()
    lowered = text.lstrip().lower()
    if lowered.startswith("[error") or lowered.startswith("[blocked"):
        return text
    if tool_name == "shell":
        return text
    return f"[error: {text}]"


def _build_llm(cfg: ModelConfig) -> LLM:
    try:
        api_key = cfg.api_key
    except EnvironmentError:
        api_key = "no-key"
    return LLM(
        base_url=cfg.base_url,
        api_key=api_key,
        model=cfg.model,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        budget_tokens=cfg.budget_tokens,
        reasoning_effort=cfg.reasoning_effort,
        cache_prompts=cfg.cache_prompts,
    )


async def _run_background(tail_node_id: str, config: Config) -> None:
    """
    Run a standalone AgentLoop on tail_node_id as a synthetic turn, discarding
    all events. Called as a detached asyncio task — never awaited by the caller.

    Usage (from a background hook):
        opening = db.add_node(parent_id=current_tail, role="user", content="...")
        asyncio.create_task(_run_background(opening.id, agent.config))
    """
    try:
        loop = AgentLoop(tail_node_id=tail_node_id, config=config, is_subagent=True)
        async for _ in loop.run(msg=None):
            pass  # events discarded
    except Exception:
        logger.exception("[background] AgentLoop on tail=%s raised", tail_node_id)


class AgentLoop:
    def __init__(self, tail_node_id: str, config: Config, *, is_subagent: bool = False) -> None:
        self.tail_node_id  = tail_node_id  # cursor — the DB node this agent runs from
        self.lane_node_id  = tail_node_id  # original lane key — never changes
        self.config        = config
        self.is_subagent   = is_subagent
        primary_mc         = config.models.get(config.llm.primary)
        self.context       = Context(
            token_limit=config.context,
            image_tokens_per_block=primary_mc.tokens_per_image if primary_mc else 280,
        )
        self.tool_handler  = ToolCallHandler()
        self._turn_count   = 0
        self.gateway       = None  # set by Lane after construction

        # Open the shared agent.db and wire the cursor in.
        self._db           = self._open_db()
        self._tail_node_id = tail_node_id
        self.context.set_db(self._db)
        self.context.set_tail(self._tail_node_id)
        self.context.set_cursor_callback(self._on_context_tail_advance)

        # Background hooks: called after AgentTextFinal with (tail_node_id, config).
        # Each hook is responsible for spawning its own detached task if needed.
        self._background_hooks: list[Callable[[str, Config], Awaitable[None]]] = []

        self._pending_background_branches: list[str] = []

        self._models: dict[str, LLM] = {
            name: _build_llm(mc)
            for name, mc in config.models.items()
            if not mc.is_embedding
        }

        self.tool_handler.register_tool(self.tool_handler.tools_search, always_on=True)
        self._load_modules()

    # ------------------------------------------------------------------
    # DB
    # ------------------------------------------------------------------

    def _open_db(self) -> ConversationDB:
        """Open (or create) agent.db in the workspace directory."""
        workspace = Path(self.config.workspace.path).expanduser().resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        return ConversationDB(workspace / "agent.db")

    def _on_context_tail_advance(self) -> None:
        """
        Called by Context.add() each time the tail node advances.
        Keeps _tail_node_id in sync with the context so callers reading
        agent._tail_node_id always see the latest cursor.
        """
        self._tail_node_id = self.context.tail_node_id
        # Propagate the updated cursor back to the public attribute.
        self.tail_node_id  = self._tail_node_id

    # ------------------------------------------------------------------
    # Public model accessor (for modules)
    # ------------------------------------------------------------------

    def get_model(self, name: str) -> LLM:
        if name in self._models:
            return self._models[name]
        primary = self.config.llm.primary
        logger.warning("get_model('%s') — not found, falling back to primary '%s'", name, primary)
        return self._models[primary]

    # ------------------------------------------------------------------
    # Background hook registry (Phase 3)
    # ------------------------------------------------------------------

    def register_background_hook(
        self, fn: Callable[[str, Config], Awaitable[None]]
    ) -> None:
        """
        Register an async hook to be called after each interactive turn
        completes (after AgentTextFinal is yielded).

        fn(tail_node_id: str, config: Config) -> Awaitable[None]

        The hook receives the tail node_id at the moment the turn finished
        and the agent's config. It is responsible for spawning its own
        detached asyncio.create_task() if it needs to run in the background
        without blocking. Hooks are called sequentially in registration order;
        exceptions are caught and logged.

        Typical usage (memory module):
            def register(agent):
                async def _memory_hook(tail_node_id, config):
                    opening = agent._db.add_node(
                        parent_id=tail_node_id, role="user",
                        content="Consolidate memory from this conversation.",
                    )
                    asyncio.create_task(_run_background(opening.id, config))
                agent.register_background_hook(_memory_hook)
        """
        self._background_hooks.append(fn)

    async def _fire_background_hooks(self, tail_node_id: str) -> None:
        """Invoke all registered background hooks after a turn completes."""
        for fn in self._background_hooks:
            try:
                await fn(tail_node_id, self.config)
            except Exception:
                logger.exception("[background] hook '%s' raised", getattr(fn, "__name__", fn))

    # ------------------------------------------------------------------
    # Module loader
    # ------------------------------------------------------------------

    def _load_modules(self) -> None:
        if not MODULES_DIR.exists():
            return
        for entry in sorted(MODULES_DIR.iterdir()):
            if not entry.is_dir():
                continue
            if not ((entry / "__main__.py").exists() or (entry / "__init__.py").exists()):
                continue
            module_name = f"modules.{entry.name}"
            try:
                for suffix in (".__main__", ""):
                    try:
                        mod = importlib.import_module(module_name + suffix)
                        if hasattr(mod, "register"):
                            break
                    except ModuleNotFoundError:
                        continue
                else:
                    logger.warning("Module '%s' has no register() — skipping", entry.name)
                    continue
                mod.register(self)
                logger.info("Loaded module '%s'", entry.name)
            except Exception:
                logger.exception("Failed to load module '%s'", entry.name)

    async def _maybe_compact_context(self) -> bool:
        compaction_cfg = getattr(self.config, "compaction", None)
        if compaction_cfg is not None and not getattr(compaction_cfg, "enabled", True):
            return False

        pretrim_tokens = int(self.context.state.get("tokens_used_pre_trim", 0) or 0)
        token_limit = int(self.config.context or 0)
        trigger_pct = float(getattr(compaction_cfg, "trigger_pct", 0.90) if compaction_cfg is not None else 0.90)
        keep_last_units = int(getattr(compaction_cfg, "keep_last_units", 4) if compaction_cfg is not None else 4)
        summary_max_chars = int(getattr(compaction_cfg, "summary_max_chars", 6000) if compaction_cfg is not None else 6000)

        if not should_compact(pretrim_tokens, token_limit, trigger_pct=trigger_pct):
            return False

        plan = build_compaction_plan(self.context.dialogue, keep_last_units=keep_last_units)
        if plan is None:
            return False

        summary = await self._generate_compact_summary(plan.to_summarize)
        if not summary:
            logger.warning(
                "[cursor=%s] compaction skipped — summary generation failed",
                self._tail_node_id,
            )
            return False

        active = self.context.compact(
            format_summary(summary, max_chars=summary_max_chars),
            preserved_tail=plan.preserved_tail,
            metadata={
                "entries_summarized": len(plan.to_summarize),
                "summarized_units": plan.summarized_units,
            },
        )
        logger.info(
            "[cursor=%s] compacted %d entry(s) into summary + %d preserved entry(s)",
            self._tail_node_id,
            len(plan.to_summarize),
            len(active) - 1,
        )
        return True

    async def _generate_compact_summary(self, entries: list[HistoryEntry]) -> str | None:
        summary_messages = (
            [{"role": "system", "content": COMPACT_SYSTEM_PROMPT}]
            + [self.context._render(entry) for entry in entries]
            + [{"role": "user", "content": COMPACT_USER_PROMPT}]
        )

        model_chain = [self.config.llm.primary] + list(self.config.llm.fallback)
        for model_name in model_chain:
            llm = self._models[model_name]
            text_chunks: list[str] = []
            error: str | None = None
            last_http_status: int | None = None

            async for llm_event in llm.stream(summary_messages, tools=None):
                if isinstance(llm_event, TextDelta):
                    text_chunks.append(llm_event.text)
                elif isinstance(llm_event, ToolCallAssembled):
                    error = "compaction summary unexpectedly returned tool calls"
                    break
                elif isinstance(llm_event, LLMError):
                    error = llm_event.message
                    if llm_event.message.startswith("HTTP "):
                        try:
                            last_http_status = int(llm_event.message.split()[1].rstrip(":"))
                        except (IndexError, ValueError):
                            pass
                    break

            if not error:
                summary = "".join(text_chunks).strip()
                if summary:
                    return summary
                error = "empty compaction summary"

            fo = self.config.llm.fallback_on
            should_fallback = fo.any_error or (
                last_http_status is not None and last_http_status in fo.http_codes
            )
            if should_fallback and model_name != model_chain[-1]:
                logger.warning(
                    "[cursor=%s] compaction model '%s' failed (%s) — trying fallback",
                    self._tail_node_id,
                    model_name,
                    error,
                )
                continue

            logger.warning(
                "[cursor=%s] compaction summary failed on model '%s': %s",
                self._tail_node_id,
                model_name,
                error,
            )
            break

        return None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        msg: InboundMessage | None = None,
        abort_event: asyncio.Event | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """
        Run one agent turn.

        msg=None  — synthetic turn: skip Stage 1, generate against current
                    context as-is (used by PUT /v1/sessions/{id}/generation).
        msg=<msg> — normal turn: add user message to context, then generate.

        After AgentTextFinal is yielded, any registered background hooks are
        fired (Phase 3). Synthetic turns do not trigger background hooks —
        they are themselves background work and should not spawn further
        background branches.
        """
        self._turn_count += 1
        is_synthetic = msg is None
        logger.debug("[cursor=%s] turn %d (synthetic=%s)", self._tail_node_id, self._turn_count, is_synthetic)

        if msg is not None:
            trace_id = msg.trace_id
            msg_id   = msg.message_id
        else:
            trace_id = str(uuid.uuid4())
            msg_id   = "synthetic"

        ev = dict(
            tail_node_id=self._tail_node_id,
            lane_node_id=self.lane_node_id,
            trace_id=trace_id,
            reply_to_message_id=msg_id,
        )

        # Stage 1: Intake (skipped for synthetic turns)
        if msg is not None:
            if msg.attachments:
                primary_cfg  = self.config.get_model_config(self.config.llm.primary)
                user_content = build_content_blocks(
                    text=msg.text,
                    attachments=msg.attachments,
                    model_cfg=primary_cfg,
                    att_cfg=self.config.attachments,
                    workspace=self.config.workspace.path,
                )
            else:
                user_content = msg.text
            self.context.add(HistoryEntry.user(user_content))

        max_cycles       = self.config.max_tool_cycles
        final_text       = ""
        streaming_active = False
        tool_result_cache: dict[str, ToolResult] = {}
        had_tool_activity = False

        compacted_this_turn = False
        for cycle in range(max_cycles):
            while True:
                # Abort check between cycles
                if abort_event and abort_event.is_set():
                    logger.info("[%s] aborted before cycle %d", self._tail_node_id, cycle)
                    yield AgentError(message="[generation aborted]", **ev)
                    return

                # Stage 2: Context Assembly
                await self.context.run_async_hooks(HOOK_PRE_ASSEMBLE_ASYNC)
                tools    = self.tool_handler.get_tool_definitions() or None
                messages = self.context.assemble(tools=tools)

                # Token budget telemetry
                tokens_used = int(self.context.state.get("tokens_used_pre_trim", 0) or 0)
                active_tokens = int(self.context.state.get("tokens_used", 0) or 0)
                token_limit = self.config.context
                token_pct   = tokens_used / token_limit if token_limit else 0
                if token_pct >= 0.95:
                    logger.warning(
                        "[cursor=%s] context at %.0f%% of token budget (%d/%d, active=%d) — consider compaction",
                        self._tail_node_id, token_pct * 100, tokens_used, token_limit, active_tokens,
                    )
                elif token_pct >= 0.80:
                    logger.info(
                        "[cursor=%s] context at %.0f%% of token budget (%d/%d, active=%d)",
                        self._tail_node_id, token_pct * 100, tokens_used, token_limit, active_tokens,
                    )

                if not compacted_this_turn and await self._maybe_compact_context():
                    compacted_this_turn = True
                    continue

                break

            # Stage 3: Inference — walk primary → fallback chain
            model_chain = [self.config.llm.primary] + list(self.config.llm.fallback)
            inference_messages = list(messages)
            empty_reply_retries = 0

            while True:
                text_chunks:      list[str]      = []
                tool_calls:       list[ToolCall] = []
                error:            str | None     = None
                streaming_active                 = False
                last_http_status: int | None     = None

                for model_name in model_chain:
                    llm              = self._models[model_name]
                    text_chunks      = []
                    tool_calls       = []
                    error            = None
                    streaming_active = False
                    last_http_status = None

                    async for llm_event in llm.stream(inference_messages, tools=tools):
                        if abort_event and abort_event.is_set():
                            logger.info("[%s] aborted mid-stream", self._tail_node_id)
                            yield AgentError(message="[generation aborted]", **ev)
                            return

                        if isinstance(llm_event, ThinkingDelta):
                            yield AgentThinkingChunk(text=llm_event.text, **ev)

                        elif isinstance(llm_event, TextDelta):
                            text_chunks.append(llm_event.text)
                            if not tool_calls:
                                streaming_active = True
                                yield AgentTextChunk(text=llm_event.text, **ev)

                        elif isinstance(llm_event, ToolCallAssembled):
                            tool_calls.append(ToolCall(
                                call_id=llm_event.call_id,
                                tool_name=llm_event.tool_name,
                                args=llm_event.args,
                            ))

                        elif isinstance(llm_event, LLMError):
                            error = llm_event.message
                            if llm_event.message.startswith("HTTP "):
                                try:
                                    last_http_status = int(llm_event.message.split()[1].rstrip(":"))
                                except (IndexError, ValueError):
                                    pass
                            break

                    if not error:
                        if model_name != self.config.llm.primary:
                            logger.info(
                                "[cursor=%s] inference succeeded on fallback model '%s'",
                                self._tail_node_id, model_name,
                            )
                            mc = self.config.models.get(model_name)
                            self.context.set_image_tokens(mc.tokens_per_image if mc else None)
                        break

                    fo = self.config.llm.fallback_on
                    should_fallback = fo.any_error or (
                        last_http_status is not None and last_http_status in fo.http_codes
                    )
                    if should_fallback and model_name != model_chain[-1]:
                        logger.warning(
                            "[cursor=%s] model '%s' failed (%s) — trying next fallback",
                            self._tail_node_id, model_name, error,
                        )
                        continue
                    break

                if error:
                    break

                response_text = "".join(text_chunks)
                if tool_calls or response_text.strip():
                    break

                if empty_reply_retries >= 1:
                    if had_tool_activity:
                        response_text = "[No final response returned after tool use.]"
                    else:
                        response_text = "[No response returned.]"
                    logger.warning(
                        "[cursor=%s] assistant returned empty final response after retry",
                        self._tail_node_id,
                    )
                    break

                empty_reply_retries += 1
                logger.warning(
                    "[cursor=%s] assistant returned empty final response — retrying once with direct-answer nudge",
                    self._tail_node_id,
                )
                inference_messages = list(messages) + [{
                    "role": "system",
                    "content": _EMPTY_REPLY_RETRY_PROMPT,
                }]

            if error:
                logger.error("[cursor=%s] LLM error (all models exhausted): %s", self._tail_node_id, error)
                yield AgentError(message=f"[LLM error: {error}]", **ev)
                return

            self.context.add(HistoryEntry.assistant(
                content=response_text,
                tool_calls=tool_calls if tool_calls else None,
            ))

            if not tool_calls:
                final_text = response_text
                break

            # Stages 4 & 5: Tool execution + result backfill
            logger.debug("[%s] cycle %d — %d tool call(s)", self._tail_node_id, cycle, len(tool_calls))
            had_tool_activity = True
            for tc in tool_calls:
                yield AgentToolCall(call_id=tc.call_id, tool_name=tc.tool_name, args=tc.args, **ev)
                result = await self._execute_tool(tc, tool_result_cache=tool_result_cache)
                self.context.add(HistoryEntry.tool_result(result))
                # For image results, inject a follow-up user message with the image_url
                # block.  OpenAI-compat servers don't support list content in tool result
                # messages, so a synthetic user turn is a shitty but ok workaround.
                if result.is_image:
                    image_content = [
                        {"type": "text",      "text": "Here is the image from the tool result:"},
                        {"type": "image_url", "image_url": {"url": f"data:{result.image_mime};base64,{result.image_b64}"}},
                    ]
                    self.context.add(HistoryEntry.user(image_content))
                # For bridge display, show a tidy label instead of raw base64.
                display_output = f"[image: {result.image_mime}]" if result.is_image else result.output
                yield AgentToolResult(
                    call_id=result.call_id,
                    tool_name=result.tool_name,
                    output=display_output,
                    is_error=result.is_error,
                    **ev,
                )

        else:
            logger.warning("[cursor=%s] hit max_tool_cycles (%d)", self._tail_node_id, max_cycles)
            final_text = final_text or "[Tool cycle limit reached.]"

        yield AgentTextFinal(text=final_text if not streaming_active else "", **ev)

        # Phase 3: fire any background branches registered this turn.
        for branch_node_id in self._pending_background_branches:
            asyncio.ensure_future(self._run_background(branch_node_id))
        self._pending_background_branches.clear()

        # Phase 3: fire background hooks after the turn completes.
        # Skipped for synthetic turns — they are already background work.
        if not is_synthetic and self._background_hooks:
            await self._fire_background_hooks(self._tail_node_id)

    # ------------------------------------------------------------------
    # Background branch runner (Phase 3)
    # ------------------------------------------------------------------

    async def _run_background(self, tail_node_id: str) -> None:
        """
        Run a synthetic agent turn on a detached branch. Events are discarded.
        The branch writes its own nodes into agent.db; the caller's cursor is
        never touched.
        """
        logger.debug("[background] starting branch tail=%s", tail_node_id)
        try:
            loop = AgentLoop(tail_node_id=tail_node_id, config=self.config, is_subagent=True)
            async for _ in loop.run(msg=None):
                pass  # discard events
            logger.debug("[background] branch complete tail=%s", tail_node_id)
        except Exception:
            logger.exception("[background] branch failed tail=%s", tail_node_id)

    def queue_background_branch(self, tail_node_id: str) -> None:
        """
        Schedule a background branch to be fired after the current turn yields
        AgentTextFinal. Called by modules (e.g. memory) during a hook.
        """
        self._pending_background_branches.append(tail_node_id)

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    async def _execute_tool(
        self,
        call: ToolCall,
        *,
        tool_result_cache: dict[str, ToolResult] | None = None,
    ) -> ToolResult:
        cache_key = None
        if tool_result_cache is not None:
            cache_key = _tool_cache_key(call)
            cached = tool_result_cache.get(cache_key)
            if cached is not None:
                logger.info(
                    "[cursor=%s] reusing cached tool result for %s",
                    self._tail_node_id, call.tool_name,
                )
                return ToolResult(
                    call_id=call.call_id,
                    tool_name=call.tool_name,
                    output=_cached_tool_result_notice(call, is_error=cached.is_error),
                    is_error=cached.is_error,
                    is_image=False,
                )

        proxy = {
            "function": {"name": call.tool_name, "arguments": call.args},
            "id": call.call_id,
        }
        result = await self.tool_handler.execute_tool_call(proxy)
        raw_output = str(result.get("result", result.get("error", "[no output]")))
        is_error = not result.get("success", False)
        if not is_error and call.tool_name == "shell":
            is_error = _looks_like_failed_shell_output(raw_output)
        if is_error:
            raw_output = _normalize_error_output(call.tool_name, raw_output)

        # --- vision unwrap ---
        # If view() returned an IMAGE_BLOCK sentinel and the primary model
        # supports vision, stash the image data in the ToolResult so the
        # caller (run()) can inject a follow-up user message containing the
        # image_url block.  OpenAI-compat servers don't support list content
        # in tool result messages, so we use a separate user turn instead.
        if not is_error and raw_output.startswith(IMAGE_BLOCK_PREFIX):
            payload = raw_output[len(IMAGE_BLOCK_PREFIX):]  # "mime;base64data"
            sep = payload.index(";")
            mime    = payload[:sep]
            b64data = payload[sep + 1:]

            primary_cfg = self.config.get_model_config(self.config.llm.primary)
            if primary_cfg.supports_vision:
                tool_result = ToolResult(
                    call_id=call.call_id,
                    tool_name=call.tool_name,
                    output=f"[image/{mime} — see attached image below]",
                    is_error=False,
                    is_image=True,
                    image_mime=mime,
                    image_b64=b64data,
                )
                if cache_key is not None and tool_result_cache is not None:
                    tool_result_cache[cache_key] = tool_result
                return tool_result
            else:
                # Model doesn't support vision — return a friendly stub.
                raw_output = (
                    f"[Image file detected ({mime}) but the current model does not "
                    "support vision. Use a vision-capable model to inspect this file.]"
                )

        tool_result = ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            output=raw_output,
            is_error=is_error,
        )
        if cache_key is not None and tool_result_cache is not None:
            tool_result_cache[cache_key] = tool_result
        return tool_result

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Clear in-memory context. Does NOT touch agent.db — the tree is
        permanent. The cursor stays at the current tail; the agent will
        continue to see prior history when it next assembles context.
        """
        self.context.clear()
        self._turn_count = 0
        logger.info("[cursor=%s] reset (in-memory only — tree intact)", self._tail_node_id)
