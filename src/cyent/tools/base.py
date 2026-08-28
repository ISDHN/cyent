"""BaseTool — declaration + local implementation for every tool."""

from __future__ import annotations

import abc
import json
from typing import Any

from cyent.core.types import ToolSchema


class BaseTool(abc.ABC):
    """A tool = OpenAI function schema + a local ``run`` implementation.

    ``run`` must return a *string* observation that will be fed back to the
    model. It must never raise past the executor (executor catches, but keep
    implementations defensive anyway).
    """

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {}  # JSON Schema

    @abc.abstractmethod
    def run(self, **kwargs: Any) -> str:
        """Execute the tool locally and return the observation text."""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name, description=self.description, parameters=self.parameters
        )

    # ------------------------------------------------------------------ #
    # Lightweight argument validation driven by the JSON Schema
    # ------------------------------------------------------------------ #
    def validate(self, args: dict[str, Any]) -> str | None:
        """Return an error string for invalid args, or None when OK.

        Deliberately light: required fields + top-level type checks only.
        """
        props = self.parameters.get("properties", {})
        for req in self.parameters.get("required", []):
            if req not in args:
                return f"missing required argument: {req!r}"
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        for key, value in args.items():
            spec = props.get(key)
            if not spec:
                continue  # unknown args are tolerated
            expected = type_map.get(spec.get("type", ""))
            if expected and not isinstance(value, expected):
                # bool is a subclass of int — reject that confusion
                if expected is int and isinstance(value, bool):
                    return f"argument {key!r} must be an integer, got bool"
                if not isinstance(value, expected):
                    return f"argument {key!r} must be {spec.get('type')}, got {type(value).__name__}"
        return None

    def format_result(self, payload: Any) -> str:
        """Serialize structured results into observation text."""
        if isinstance(payload, str):
            return payload
        return json.dumps(payload, ensure_ascii=False, indent=2, default=str)
