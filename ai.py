"""
ai.py — Async OpenAI-compatible LLM and Embedder clients.
Streams SSE, assembles tool call deltas, yields typed events.
Imports only aiohttp and stdlib. No internal project imports.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import AsyncIterator, Any
import aiohttp
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Yield types
# ---------------------------------------------------------------------------

@dataclass
class TextDelta:
    text: str

@dataclass
class ToolCallAssembled:
    """Emitted once per tool call, after all argument chunks are assembled."""
    call_id:   str
    tool_name: str
    args:      dict[str, Any]

@dataclass
class LLMError:
    message: str


LLMEvent = TextDelta | ToolCallAssembled | LLMError


# ---------------------------------------------------------------------------
# Chat client
# ---------------------------------------------------------------------------

class LLM:
    """
    Async OpenAI-compatible streaming client.
    Works with Anthropic (via OpenAI-compat endpoint), OpenAI, OpenRouter,
    LM Studio, Ollama, or any base_url that speaks /v1/chat/completions.
    """

    def __init__(
        self,
        base_url:    str,
        api_key:     str,
        model:       str,
        max_tokens:  int   = 2048,
        temperature: float = 0.7,
        timeout:     int   = 60,
    ) -> None:
        self.model       = model
        self.endpoint    = f"{base_url.rstrip('/')}/chat/completions"
        self.api_key     = api_key
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self.timeout     = aiohttp.ClientTimeout(total=timeout)

    async def stream(
        self,
        messages: list[dict],
        tools:    list[dict] | None = None,
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
        tools:    list[dict] | None = None,
    ) -> AsyncIterator[LLMEvent]:
        payload: dict[str, Any] = {
            "model":       self.model,
            "messages":    messages,
            "temperature": self.temperature,
            "max_tokens":  self.max_tokens,
            "stream":      True,
        }
        if tools:
            payload["tools"] = tools

        headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        # Accumulate tool call fragments keyed by index
        # { index: {"id": str, "name": str, "args_buf": str} }
        tool_buf: dict[int, dict] = {}

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(
                    self.endpoint, headers=headers, json=payload
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        yield LLMError(f"HTTP {resp.status}: {body}")
                        return

                    async for raw in resp.content:
                        line = raw.decode().strip()
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        choices = data.get("choices")
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})

                        # Text content
                        if text := delta.get("content"):
                            yield TextDelta(text=text)

                        # Tool call fragments — assemble before yielding
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

                    # Stream done — emit assembled tool calls
                    for buf in tool_buf.values():
                        try:
                            args = json.loads(buf["args_buf"] or "{}")
                        except json.JSONDecodeError:
                            args = {"_raw": buf["args_buf"]}
                        yield ToolCallAssembled(
                            call_id=buf["id"],
                            tool_name=buf["name"],
                            args=args,
                        )

        except aiohttp.ClientConnectionError:
            raise  # tenacity will retry on this
        except Exception as e:
            yield LLMError(str(e))


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
        base_url:   str,
        api_key:    str,
        model:      str,
        batch_size: int = 32,
        timeout:    int = 60,
    ) -> None:
        self.model      = model
        self.endpoint   = f"{base_url.rstrip('/')}/embeddings"
        self.api_key    = api_key
        self.batch_size = batch_size
        self.timeout    = aiohttp.ClientTimeout(total=timeout)

    @classmethod
    def from_config(cls, cfg: "ModelConfig", batch_size: int = 32, timeout: int = 60) -> "Embedder":  # noqa: F821
        """Build an Embedder from a ModelConfig with kind='embedding'."""
        api_key = cfg.api_key  # resolves from env or returns "" for N/A
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
            "Content-Type":  "application/json",
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

        # Sort by index to guarantee order matches input regardless of server behaviour
        items = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in items]
