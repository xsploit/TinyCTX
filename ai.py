"""
ai.py — Async OpenAI-compatible LLM and Embedder clients.
Streams SSE, assembles tool call deltas, yields typed events.
Imports only aiohttp and stdlib. No internal project imports.
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, AsyncIterator

import aiohttp
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Yield types
# ---------------------------------------------------------------------------


@dataclass
class TextDelta:
    text: str


@dataclass
class ThinkingDelta:
    text: str


@dataclass
class ToolCallAssembled:
    """Emitted once per tool call, after all argument chunks are assembled."""

    call_id: str
    tool_name: str
    args: dict[str, Any]


@dataclass
class LLMError:
    message: str


LLMEvent = TextDelta | ThinkingDelta | ToolCallAssembled | LLMError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _inject_cache_control(messages: list[dict]) -> list[dict]:
    """
    Return a shallow copy of messages with Anthropic prompt-caching headers
    injected on the last system message.

    The last system message's content is converted to a content-block list
    if it isn't already one, and a cache_control block is appended:
        {"type": "text", "text": "...", "cache_control": {"type": "ephemeral"}}

    If no system message is present, messages are returned unchanged.
    """

    last_sys = next(
        (i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "system"),
        None,
    )
    if last_sys is None:
        return messages

    messages = list(messages)
    msg = dict(messages[last_sys])
    content = msg.get("content", "")

    if isinstance(content, str):
        msg["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
    elif isinstance(content, list) and content:
        blocks = list(content)
        last_block = dict(blocks[-1])
        last_block["cache_control"] = {"type": "ephemeral"}
        blocks[-1] = last_block
        msg["content"] = blocks

    messages[last_sys] = msg
    return messages


def _coerce_slot_id(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        raw = value.strip()
        if raw and raw.lstrip("-").isdigit():
            return int(raw)
    return None


def _extract_slot_id(payload: dict[str, Any]) -> int | None:
    for container in (payload, payload.get("__verbose")):
        if not isinstance(container, dict):
            continue
        for key in ("id_slot", "slot_id"):
            slot_id = _coerce_slot_id(container.get(key))
            if slot_id is not None and slot_id >= 0:
                return slot_id
    return None


async def _iter_sse_events(response) -> AsyncIterator[tuple[str | None, dict[str, Any] | None]]:
    """
    Parse SSE frames from an aiohttp response.

    Yields `(event_name, data_dict)` and returns `(None, None)` only for ignored
    frames. `[DONE]` is surfaced as `("done", None)`.
    """

    buffer = ""
    event_name: str | None = None
    data_lines: list[str] = []

    def _flush_data(current_event: str | None, current_data: list[str]) -> tuple[str | None, dict[str, Any] | None] | None:
        if not current_data:
            return None
        data_str = "\n".join(current_data)
        if data_str == "[DONE]":
            return current_event or "done", None
        try:
            return current_event, json.loads(data_str)
        except json.JSONDecodeError:
            return None

    async for raw in response.content:
        buffer += raw.decode("utf-8", errors="ignore")

        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.rstrip("\r")

            if not line:
                if not data_lines:
                    event_name = None
                    continue

                data_str = "\n".join(data_lines)
                data_lines = []

                if data_str == "[DONE]":
                    yield event_name or "done", None
                else:
                    try:
                        yield event_name, json.loads(data_str)
                    except json.JSONDecodeError:
                        pass
                event_name = None
                continue

            if line.startswith("event:"):
                event_name = line[6:].strip()
                continue

            if line.startswith("data:"):
                if event_name is None and data_lines:
                    flushed = _flush_data(event_name, data_lines)
                    if flushed is not None:
                        yield flushed
                    data_lines = []
                data_lines.append(line[5:].lstrip())

    if data_lines:
        flushed = _flush_data(event_name, data_lines)
        if flushed is not None:
            yield flushed


def _stringify_block(block: Any) -> str:
    if isinstance(block, dict):
        if block.get("type") == "text":
            return str(block.get("text", ""))
        return json.dumps(block, ensure_ascii=False)
    return str(block)


def _content_to_input_blocks(content: str | list) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]

    blocks: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            text = str(block).strip()
            if text:
                blocks.append({"type": "input_text", "text": text})
            continue

        block_type = block.get("type")
        if block_type == "text":
            text = str(block.get("text", ""))
            blocks.append({"type": "input_text", "text": text})
        elif block_type == "image_url":
            image_url = block.get("image_url", {})
            url = image_url.get("url") if isinstance(image_url, dict) else None
            if url:
                blocks.append({"type": "input_image", "image_url": url})
        else:
            raw = _stringify_block(block).strip()
            if raw:
                blocks.append({"type": "input_text", "text": raw})

    return blocks or [{"type": "input_text", "text": ""}]


def _content_to_output_blocks(content: str | list) -> list[dict[str, Any]]:
    if isinstance(content, str):
        if content == "":
            return []
        return [{"type": "output_text", "text": content}]

    blocks: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            text = str(block).strip()
            if text:
                blocks.append({"type": "output_text", "text": text})
            continue

        if block.get("type") == "text":
            text = str(block.get("text", ""))
            if text:
                blocks.append({"type": "output_text", "text": text})
        else:
            raw = _stringify_block(block).strip()
            if raw:
                blocks.append({"type": "output_text", "text": raw})

    return blocks


def _content_to_tool_output(content: str | list) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content

    blocks: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            text = str(block).strip()
            if text:
                blocks.append({"type": "input_text", "text": text})
            continue

        if block.get("type") == "text":
            text = str(block.get("text", ""))
            if text:
                blocks.append({"type": "input_text", "text": text})
        else:
            raw = _stringify_block(block).strip()
            if raw:
                blocks.append({"type": "input_text", "text": raw})

    return blocks or ""


def _messages_to_responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    for message in messages:
        role = message.get("role")
        content = message.get("content", "")

        if role in {"user", "system", "developer"}:
            items.append({
                "role": role,
                "content": _content_to_input_blocks(content),
            })
            continue

        if role == "assistant":
            reasoning = message.get("reasoning_content")
            if reasoning:
                items.append({
                    "type": "reasoning",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": str(reasoning)}],
                })

            output_blocks = _content_to_output_blocks(content)
            if output_blocks:
                items.append({
                    "role": "assistant",
                    "type": "message",
                    "content": output_blocks,
                })

            for tool_call in message.get("tool_calls", []) or []:
                function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
                arguments = function.get("arguments", "{}")
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                items.append({
                    "type": "function_call",
                    "call_id": tool_call.get("id", ""),
                    "name": function.get("name", ""),
                    "arguments": arguments,
                })
            continue

        if role == "tool":
            items.append({
                "type": "function_call_output",
                "call_id": message.get("tool_call_id", ""),
                "output": _content_to_tool_output(content),
            })

    return items


def _tools_to_responses_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    responses_tools: list[dict[str, Any]] = []

    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") != "function":
            continue

        # Accept both Chat Completions-style function defs:
        #   {"type":"function","function":{...}}
        # and native Responses-style defs:
        #   {"type":"function","name":"...","parameters":{...}}
        fn = tool.get("function", tool)
        if not isinstance(fn, dict):
            continue

        responses_tool = {
            "type": "function",
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
        }
        if "strict" in fn:
            responses_tool["strict"] = fn["strict"]
        responses_tools.append(responses_tool)

    return responses_tools


def _normalize_response_output_items(output: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []

    for item in output:
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")
        if item_type == "message" and item.get("role") == "assistant":
            content: list[dict[str, Any]] = []
            for part in item.get("content", []) or []:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "output_text":
                    content.append({
                        "type": "output_text",
                        "text": str(part.get("text", "")),
                    })
            if content:
                normalized.append({
                    "role": "assistant",
                    "type": "message",
                    "content": content,
                })
            continue

        if item_type == "function_call":
            normalized.append({
                "type": "function_call",
                "call_id": item.get("call_id", ""),
                "name": item.get("name", ""),
                "arguments": item.get("arguments", ""),
            })

    return normalized


def _items_start_with(items: list[dict[str, Any]], prefix: list[dict[str, Any]]) -> bool:
    if len(prefix) > len(items):
        return False
    return items[: len(prefix)] == prefix


def _parse_tool_args(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {"_raw": raw}


# ---------------------------------------------------------------------------
# Chat client
# ---------------------------------------------------------------------------


class LLM:
    """
    Async OpenAI-compatible streaming client.
    Supports `/v1/chat/completions` and `/v1/responses`.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        timeout: int = 60,
        budget_tokens: int | None = None,
        reasoning_effort: str | None = None,
        cache_prompts: bool = False,
        llama_cpp_cache_prompt: bool = False,
        llama_cpp_sticky_slots: bool = False,
        llama_cpp_slot_id: int | None = None,
        responses_previous_response_id: bool | None = None,
        kind: str = "chat",
    ) -> None:
        self.model = model
        self.kind = kind.lower()
        if self.kind not in {"chat", "responses"}:
            raise ValueError(f"Unsupported LLM kind '{kind}'")
        suffix = "/responses" if self.kind == "responses" else "/chat/completions"
        self.endpoint = f"{base_url.rstrip('/')}{suffix}"
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.budget_tokens = budget_tokens
        self.reasoning_effort = reasoning_effort
        self.cache_prompts = cache_prompts
        self.llama_cpp_cache_prompt = llama_cpp_cache_prompt
        self.llama_cpp_sticky_slots = llama_cpp_sticky_slots
        self.llama_cpp_slot_id = llama_cpp_slot_id
        self.responses_previous_response_id = responses_previous_response_id
        self._sticky_slot_id: int | None = None
        self._last_response_id: str | None = None
        self._last_chain_items: list[dict[str, Any]] | None = None

    def _request_slot_id(self) -> int | None:
        if self.llama_cpp_slot_id is not None:
            return self.llama_cpp_slot_id
        if self.llama_cpp_sticky_slots:
            return self._sticky_slot_id
        return None

    def _capture_slot_id(self, payload: dict[str, Any]) -> None:
        if self.llama_cpp_slot_id is not None or not self.llama_cpp_sticky_slots:
            return
        slot_id = _extract_slot_id(payload)
        if slot_id is not None:
            self._sticky_slot_id = slot_id

    def reset(self) -> None:
        self._sticky_slot_id = None
        self._last_response_id = None
        self._last_chain_items = None

    def _supports_responses_chaining(self) -> bool:
        """
        OpenAI-style Responses chaining requires `previous_response_id`.

        llama.cpp's `/v1/responses` rejects that field today, so when any
        llama.cpp-specific compatibility knob is enabled we fall back to
        sending the full input history on each request.
        """
        if self.responses_previous_response_id is not None:
            return self.responses_previous_response_id
        return not (
            self.llama_cpp_cache_prompt
            or self.llama_cpp_sticky_slots
            or self.llama_cpp_slot_id is not None
        )

    async def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[LLMEvent]:
        """
        Stream a completion. Yields TextDelta, ToolCallAssembled, or LLMError.
        Retries on transient connection errors (up to 3 attempts, exponential backoff).
        Tool call argument chunks are assembled before yielding — callers
        always receive complete, parseable args dicts.
        """

        try:
            async for event in self._stream_with_retry(messages, tools):
                yield event
        except aiohttp.ClientConnectionError as e:
            yield LLMError(f"Connection failed after retries: {e}")

    @retry(
        retry=retry_if_exception_type(aiohttp.ClientConnectionError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=False,
    )
    async def _stream_with_retry(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[LLMEvent]:
        if self.kind == "responses":
            async for event in self._stream_responses(messages, tools):
                yield event
            return

        async for event in self._stream_chat(messages, tools):
            yield event

    async def _stream_chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[LLMEvent]:
        if self.cache_prompts:
            messages = _inject_cache_control(messages)

        temperature = self.temperature
        if self.budget_tokens is not None and temperature != 1.0:
            logger.warning(
                "budget_tokens requires temperature=1; overriding %.2f → 1.0 for model %s",
                temperature,
                self.model,
            )
            temperature = 1.0

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
        if self.budget_tokens is not None:
            payload["thinking"] = {"type": "enabled", "budget_tokens": self.budget_tokens}
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.llama_cpp_cache_prompt:
            payload["cache_prompt"] = True
        if (slot_id := self._request_slot_id()) is not None:
            payload["id_slot"] = slot_id
            payload["slot_id"] = slot_id

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        tool_buf: dict[int, dict[str, Any]] = {}

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(self.endpoint, headers=headers, json=payload) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        yield LLMError(f"HTTP {resp.status}: {body}")
                        return

                    async for _, data in _iter_sse_events(resp):
                        if data is None:
                            break

                        self._capture_slot_id(data)

                        choices = data.get("choices")
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})

                        if reasoning := delta.get("reasoning_content"):
                            yield ThinkingDelta(text=reasoning)

                        if text := delta.get("content"):
                            yield TextDelta(text=text)

                        for tc in delta.get("tool_calls", []):
                            idx = tc.get("index", 0)
                            if idx not in tool_buf:
                                tool_buf[idx] = {"id": "", "name": "", "args_buf": ""}
                            buf = tool_buf[idx]
                            if tc.get("id"):
                                buf["id"] = tc["id"]
                            fn = tc.get("function", {})
                            if fn.get("name"):
                                buf["name"] = fn["name"]
                            buf["args_buf"] += fn.get("arguments", "")

                    for buf in tool_buf.values():
                        yield ToolCallAssembled(
                            call_id=buf["id"],
                            tool_name=buf["name"],
                            args=_parse_tool_args(buf["args_buf"]),
                        )

        except aiohttp.ClientConnectionError:
            raise
        except Exception as e:
            yield LLMError(str(e))

    async def _stream_responses(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[LLMEvent]:
        full_input_items = _messages_to_responses_input(messages)
        request_items = full_input_items
        previous_response_id: str | None = None

        if (
            self._supports_responses_chaining()
            and
            self._last_response_id
            and self._last_chain_items
            and _items_start_with(full_input_items, self._last_chain_items)
            and len(full_input_items) > len(self._last_chain_items)
        ):
            previous_response_id = self._last_response_id
            request_items = full_input_items[len(self._last_chain_items) :]

        payload: dict[str, Any] = {
            "model": self.model,
            "input": request_items,
            "max_output_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": True,
        }
        if previous_response_id is not None:
            payload["previous_response_id"] = previous_response_id
        if tools:
            payload["tools"] = _tools_to_responses_tools(tools)
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.llama_cpp_cache_prompt:
            payload["cache_prompt"] = True
        if (slot_id := self._request_slot_id()) is not None:
            payload["id_slot"] = slot_id
            payload["slot_id"] = slot_id

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        tool_buf: dict[str, dict[str, Any]] = {}
        emitted_tool_ids: set[str] = set()
        completed_response: dict[str, Any] | None = None

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(self.endpoint, headers=headers, json=payload) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        yield LLMError(f"HTTP {resp.status}: {body}")
                        return

                    async for _, data in _iter_sse_events(resp):
                        if data is None:
                            break

                        self._capture_slot_id(data)
                        event_type = data.get("type", "")

                        if event_type == "response.reasoning_text.delta":
                            delta = data.get("delta")
                            if delta:
                                yield ThinkingDelta(text=delta)
                            continue

                        if event_type == "response.output_text.delta":
                            delta = data.get("delta")
                            if delta:
                                yield TextDelta(text=delta)
                            continue

                        if event_type == "response.output_item.added":
                            item = data.get("item", {})
                            if item.get("type") == "function_call":
                                call_id = item.get("call_id", "")
                                tool_buf.setdefault(call_id, {
                                    "call_id": call_id,
                                    "name": item.get("name", ""),
                                    "args_buf": item.get("arguments", ""),
                                })
                            continue

                        if event_type == "response.function_call_arguments.delta":
                            item_id = data.get("item_id", "")
                            if not item_id:
                                continue
                            buf = tool_buf.setdefault(item_id, {
                                "call_id": item_id,
                                "name": "",
                                "args_buf": "",
                            })
                            buf["args_buf"] += data.get("delta", "")
                            continue

                        if event_type == "response.output_item.done":
                            item = data.get("item", {})
                            if item.get("type") != "function_call":
                                continue

                            call_id = item.get("call_id", "")
                            if not call_id:
                                continue

                            buf = tool_buf.setdefault(call_id, {
                                "call_id": call_id,
                                "name": "",
                                "args_buf": "",
                            })
                            if item.get("name"):
                                buf["name"] = item["name"]
                            if item.get("arguments"):
                                buf["args_buf"] = item["arguments"]

                            if call_id not in emitted_tool_ids:
                                emitted_tool_ids.add(call_id)
                                yield ToolCallAssembled(
                                    call_id=call_id,
                                    tool_name=buf["name"],
                                    args=_parse_tool_args(buf["args_buf"]),
                                )
                            continue

                        if event_type == "response.completed":
                            response_obj = data.get("response")
                            if isinstance(response_obj, dict):
                                completed_response = response_obj

                    for call_id, buf in tool_buf.items():
                        if call_id in emitted_tool_ids:
                            continue
                        emitted_tool_ids.add(call_id)
                        yield ToolCallAssembled(
                            call_id=call_id,
                            tool_name=buf["name"],
                            args=_parse_tool_args(buf["args_buf"]),
                        )

        except aiohttp.ClientConnectionError:
            raise
        except Exception as e:
            yield LLMError(str(e))
            return

        if not isinstance(completed_response, dict):
            return

        response_id = completed_response.get("id")
        output_items = completed_response.get("output", [])
        if not isinstance(response_id, str) or not isinstance(output_items, list):
            return

        self._last_response_id = response_id
        self._last_chain_items = deepcopy(
            full_input_items + _normalize_response_output_items(output_items)
        )


# ---------------------------------------------------------------------------
# Embedding client
# ---------------------------------------------------------------------------


class Embedder:
    """
    Async OpenAI-compatible embedding client.
    Calls /v1/embeddings and returns float vectors.

    Works with any server that speaks the OpenAI embeddings API:
      - OpenAI          base_url = https://api.openai.com/v1
      - llama-swap      base_url = http://localhost:8080/v1
      - Ollama          base_url = http://localhost:11434/v1
      - LM Studio       base_url = http://localhost:1234/v1

    Usage:
        embedder = Embedder.from_config(agent.config.get_embedding_model("embed"))
        vectors = await embedder.embed(["hello", "world"])
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        batch_size: int = 32,
        timeout: int = 60,
    ) -> None:
        self.model = model
        self.endpoint = f"{base_url.rstrip('/')}/embeddings"
        self.api_key = api_key
        self.batch_size = batch_size
        self.timeout = aiohttp.ClientTimeout(total=timeout)

    @classmethod
    def from_config(cls, cfg: "ModelConfig", batch_size: int = 32, timeout: int = 60) -> "Embedder":  # noqa: F821
        """Build an Embedder from a ModelConfig with kind='embedding'."""

        api_key = cfg.api_key
        return cls(
            base_url=cfg.base_url,
            api_key=api_key,
            model=cfg.model,
            batch_size=batch_size,
            timeout=timeout,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of strings. Returns one float vector per input text,
        in the same order as the input. Batches automatically.

        Raises RuntimeError on API error.
        """

        if not texts:
            return []

        results: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            results.extend(await self._call(batch))
        return results

    async def embed_one(self, text: str) -> list[float]:
        """Convenience wrapper — embed a single string."""

        vecs = await self.embed([text])
        return vecs[0]

    async def _call(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self.model, "input": texts}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(
                    self.endpoint, headers=headers, json=payload
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        raise RuntimeError(f"Embedding API HTTP {resp.status}: {body}")
                    data = await resp.json()
        except aiohttp.ClientConnectionError as e:
            raise RuntimeError(f"Embedding API connection failed: {e}") from e

        items = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in items]
