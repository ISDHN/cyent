"""Core data structures aligned with the OpenAI ``chat.completions`` protocol.

Only OpenAI semantics are modeled here (roles: system/user/assistant/tool,
``tool_calls`` requests and ``tool`` result messages). There is deliberately
no Anthropic branch anywhere in this project.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(slots=True)
class ToolCall:
    """One tool invocation requested by the model.

    Mirrors OpenAI's ``chat.completions`` ``tool_calls[i]`` entry:
    ``id`` + ``function.name`` + ``function.arguments`` (JSON string).
    ``arguments`` holds the *parsed* arguments object; ``raw_arguments``
    keeps the original string for debugging / re-parsing.
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_arguments: str = ""

    def to_openai(self) -> dict[str, Any]:
        """Serialize back to the OpenAI wire format."""
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.raw_arguments
                or json.dumps(self.arguments, ensure_ascii=False),
            },
        }

    @classmethod
    def from_openai(cls, item: dict[str, Any]) -> ToolCall:
        """Build from an OpenAI wire-format dict (arguments kept raw)."""
        fn = item.get("function") or {}
        raw = fn.get("arguments") or ""
        return cls(
            id=item.get("id") or "",
            name=fn.get("name") or "",
            raw_arguments=(
                raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
            ),
        )


@dataclass(slots=True)
class Message:
    """Unified message carrier for all four OpenAI roles.

    - ``system`` / ``user``: only ``content`` matters.
    - ``assistant``: ``content`` (may be None) and/or ``tool_calls``.
    - ``tool``: ``tool_call_id`` + ``content`` (the observation text).
    """

    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_openai(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": self.role}
        if self.role == "assistant":
            # content must be present (may be empty string) when tool_calls exist
            msg["content"] = self.content if self.content is not None else None
            if self.tool_calls:
                msg["tool_calls"] = [tc.to_openai() for tc in self.tool_calls]
        elif self.role == "tool":
            msg["content"] = self.content if self.content is not None else ""
            msg["tool_call_id"] = self.tool_call_id or ""
            if self.name:
                msg["name"] = self.name
        else:
            msg["content"] = self.content if self.content is not None else ""
        return msg

    @classmethod
    def from_openai(cls, data: dict[str, Any]) -> Message:
        tool_calls = None
        if data.get("tool_calls"):
            tool_calls = [ToolCall.from_openai(tc) for tc in data["tool_calls"]]
        return cls(
            role=data.get("role", "user"),
            content=data.get("content"),
            tool_calls=tool_calls,
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
        )

    # ------------------------------------------------------------------ #
    # Session-archive serialization (JSONL rows; see core/session.py)
    # ------------------------------------------------------------------ #
    def to_archive(self) -> dict[str, Any]:
        """Wire format plus the archive row marker."""
        return {"type": "message", **self.to_openai()}

    @classmethod
    def from_archive(cls, data: dict[str, Any]) -> Message:
        """Inverse of ``to_archive`` (ignores the row marker)."""
        return cls.from_openai(data)

    # ------------------------------------------------------------------ #
    # Convenience constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role="user", content=content)

    @classmethod
    def assistant(
        cls,
        content: str | None = None,
        tool_calls: list[ToolCall] | None = None,
    ) -> Message:
        return cls(role="assistant", content=content, tool_calls=tool_calls)

    @classmethod
    def tool_result(
        cls, tool_call_id: str, content: str, name: str | None = None
    ) -> Message:
        return cls(role="tool", content=content, tool_call_id=tool_call_id, name=name)

    # ------------------------------------------------------------------ #
    # Introspection helpers
    # ------------------------------------------------------------------ #
    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def text(self) -> str:
        """Best-effort plain text of this message (for logs / estimates)."""
        if self.content:
            return self.content
        if self.tool_calls:
            parts = [f"{tc.name}({tc.raw_arguments})" for tc in self.tool_calls]
            return " ".join(parts)
        return ""

    def approx_tokens(self) -> int:
        """Cheap token estimate (~4 chars/token, plus per-message overhead)."""
        return 4 + len(self.text()) // 4


@dataclass(slots=True)
class ToolSchema:
    """Tool declaration, serializable to OpenAI's function format."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema object

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(slots=True)
class Usage:
    """Token usage reported by the API (0 when unknown)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_openai(cls, usage: Any) -> Usage:
        if usage is None:
            return cls()
        get = getattr(usage, "model_dump", None)
        data: dict[str, Any] = get() if callable(get) else usage
        if isinstance(data, dict):
            return cls(
                prompt_tokens=int(data.get("prompt_tokens") or 0),
                completion_tokens=int(data.get("completion_tokens") or 0),
                total_tokens=int(data.get("total_tokens") or 0),
            )
        # plain object with attributes
        return cls(
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
        )


@dataclass(slots=True)
class ChatResult:
    """Normalized result of one model call."""

    message: Message
    finish_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    raw: Any = None  # original SDK response (non-stream) for debugging
