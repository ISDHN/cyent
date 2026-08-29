"""LLM client layer — thin wrapper over the ``openai`` SDK.

Supports any OpenAI-compatible endpoint (gateway / DeepSeek / local vLLM).
Provides streaming and non-streaming chat; streaming merges split
``tool_calls`` deltas back into complete calls by index.
"""

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from cyent.config.env import Settings
from cyent.core.types import ChatResult, Message, ToolCall, ToolSchema, Usage
from cyent.utils.errors import LLMError, wrap_openai_error

log = logging.getLogger("cyent.llm")


@dataclass(slots=True)
class StreamEvent:
    """One incremental event from a streaming call, consumed by the CLI."""

    kind: str  # "text_delta" | "thinking_delta" | "tool_call_delta" | "done"
    text: str = ""
    tool_index: int | None = None
    tool_id: str = ""
    tool_name: str = ""
    tool_args_delta: str = ""


class LLMClient:
    """Unified chat/stream entry points over the OpenAI SDK."""

    def __init__(self) -> None:
        settings = Settings.get()
        self._settings = settings
        self._client = OpenAI(
            base_url=settings.base_url,
            api_key=settings.api_key or "missing",
        )

    # ------------------------------------------------------------------ #
    # Non-streaming
    # ------------------------------------------------------------------ #
    def chat(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> ChatResult:
        """One-shot chat completion; returns a normalized ChatResult."""
        kwargs: dict[str, Any] = {
            "model": self._settings.model,
            "messages": [m.to_openai() for m in messages],
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = [t.to_openai() for t in tools]
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        log.debug(
            "chat() request: model=%s, messages=%d, tools=%d",
            self._settings.model,
            len(messages),
            len(tools or []),
        )
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise wrap_openai_error(exc)

        choice = resp.choices[0]
        msg = Message.from_openai(choice.message.model_dump(exclude_none=True))
        result = ChatResult(
            message=msg,
            finish_reason=choice.finish_reason,
            usage=Usage.from_openai(resp.usage),
            raw=resp,
        )
        log.info(
            "chat() done: finish=%s, usage=%s",
            result.finish_reason,
            result.usage.total_tokens,
        )
        return result

    # ------------------------------------------------------------------ #
    # Streaming
    # ------------------------------------------------------------------ #
    def chat_stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> tuple[Iterator[StreamEvent], CallableResult]:
        """Streaming chat completion.

        Returns an iterator of StreamEvent plus a ``CallableResult`` holder;
        after the stream is fully consumed, ``holder.result`` contains the
        assembled ChatResult (with merged tool_calls and usage).
        """
        kwargs: dict[str, Any] = {
            "model": self._settings.model,
            "messages": [m.to_openai() for m in messages],
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = [t.to_openai() for t in tools]
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        log.debug(
            "chat_stream() request: model=%s, messages=%d",
            self._settings.model,
            len(messages),
        )
        try:
            stream = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise wrap_openai_error(exc)

        holder = CallableResult()
        return self._consume_stream(stream, holder), holder

    def _consume_stream(
        self, stream: Any, holder: CallableResult
    ) -> Iterator[StreamEvent]:
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        # tool_calls deltas merged by index: {index: {"id":..., "name":..., "args": [...]}}
        tool_acc: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        usage = Usage()

        try:
            for chunk in stream:
                if getattr(chunk, "usage", None):
                    usage = Usage.from_openai(chunk.usage)
                if not getattr(chunk, "choices", None):
                    continue
                choice = chunk.choices[0]
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                delta = choice.delta
                if delta is None:
                    continue

                # reasoning delta (OpenAI-compatible reasoning models expose
                # it as delta.reasoning_content / delta.reasoning); streamed
                # separately so the CLI can dim it. Not kept in the message.
                reasoning_text = getattr(delta, "reasoning_content", None) or getattr(
                    delta, "reasoning", None
                )
                if reasoning_text:
                    thinking_parts.append(reasoning_text)
                    yield StreamEvent(kind="thinking_delta", text=reasoning_text)

                # text delta
                if delta.content:
                    text_parts.append(delta.content)
                    yield StreamEvent(kind="text_delta", text=delta.content)

                # tool_calls deltas — merge by index
                for tc in delta.tool_calls or []:
                    idx = tc.index or 0
                    acc = tool_acc.setdefault(idx, {"id": "", "name": "", "args": []})
                    if tc.id:
                        acc["id"] = tc.id
                    fn = tc.function
                    if fn is not None:
                        if fn.name:
                            acc["name"] += fn.name
                        if fn.arguments:
                            acc["args"].append(fn.arguments)
                            yield StreamEvent(
                                kind="tool_call_delta",
                                tool_index=idx,
                                tool_id=acc["id"],
                                tool_name=acc["name"],
                                tool_args_delta=fn.arguments,
                            )
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()

        tool_calls = [
            ToolCall(
                id=acc["id"] or f"call_{idx}",
                name=acc["name"],
                raw_arguments="".join(acc["args"]),
            )
            for idx, acc in sorted(tool_acc.items())
        ]
        message = Message.assistant(
            content="".join(text_parts) or None, tool_calls=tool_calls or None
        )
        holder.result = ChatResult(
            message=message,
            finish_reason=finish_reason,
            usage=usage,
        )
        log.info(
            "chat_stream() done: finish=%s, tool_calls=%d, thinking=%d chars, usage=%s",
            finish_reason,
            len(tool_calls),
            len("".join(thinking_parts)),
            usage.total_tokens,
        )
        yield StreamEvent(kind="done")


@dataclass
class CallableResult:
    """Mutable holder so stream consumers can fetch the final ChatResult."""

    result: ChatResult | None = field(default=None)

    def get(self) -> ChatResult:
        if self.result is None:
            raise LLMError("Stream not fully consumed yet.")
        return self.result


def dump_json(obj: Any) -> str:
    """Compact JSON dump used for logs."""
    return json.dumps(obj, ensure_ascii=False, default=str)
