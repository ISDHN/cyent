"""Tool executor: registry, validation, dispatch, isolation, timeouts.

Every tool call is wrapped: exceptions never propagate to the engine — they
become readable observation text fed back to the model. Outputs are redacted
before being returned.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from cyent.core.types import ToolCall, ToolSchema
from cyent.tools.base import BaseTool
from cyent.utils.errors import ToolError, ToolValidationError
from cyent.utils.redact import redact

log = logging.getLogger("cyent.executor")


class ToolExecutor:
    """Holds the tool registry and executes model-requested tool calls."""

    def __init__(self, tools: list[BaseTool], secrets: list[str] | None = None) -> None:
        self._tools: dict[str, BaseTool] = {}
        for tool in tools:
            if not tool.name:
                raise ToolError(f"tool {tool.__class__.__name__} has no name")
            if tool.name in self._tools:
                raise ToolError(f"duplicate tool name: {tool.name}")
            self._tools[tool.name] = tool
        self._secrets = secrets or []

    # ------------------------------------------------------------------ #
    # Registry access
    # ------------------------------------------------------------------ #
    @property
    def tool_names(self) -> list[str]:
        return list(self._tools)

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def schemas(self) -> list[ToolSchema]:
        """All tool declarations, ready to send to the model."""
        return [t.schema for t in self._tools.values()]

    def schemas_json(self) -> str:
        """Compact JSON of all schemas (for logs / system prompt)."""
        return json.dumps(
            [s.to_openai() for s in self.schemas()], ensure_ascii=False, indent=2
        )

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #
    def execute(self, call: ToolCall) -> str:
        """Execute one ToolCall and return the observation text (never raises)."""
        name = call.name
        started = time.monotonic()
        log.info(
            "tool start: %s args=%s", name, redact(call.raw_arguments, self._secrets)
        )

        try:
            observation = self._dispatch(call)
        except Exception as exc:  # noqa: BLE001 — isolation is the whole point
            log.exception("tool %s crashed", name)
            observation = (
                f"ERROR: tool {name!r} crashed: {exc.__class__.__name__}: {exc}"
            )

        elapsed = time.monotonic() - started
        observation = redact(observation, self._secrets)
        log.info("tool done: %s in %.2fs, %d chars", name, elapsed, len(observation))
        return observation

    def _dispatch(self, call: ToolCall) -> str:
        tool = self._tools.get(call.name)
        if tool is None:
            known = ", ".join(sorted(self._tools)) or "(none)"
            return f"ERROR: unknown tool {call.name!r}. Available tools: {known}"

        # Validation (schema-driven, light)
        err = tool.validate(call.arguments)
        if err:
            raise ToolValidationError(f"invalid arguments for {call.name!r}: {err}")

        try:
            result = tool.run(**call.arguments)
        except TypeError as exc:
            # unexpected kwarg or signature mismatch -> readable retry hint
            raise ToolValidationError(
                f"{call.name!r} rejected arguments: {exc}"
            ) from exc
        except ToolError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ToolError(
                f"{call.name!r} failed: {exc.__class__.__name__}: {exc}"
            ) from exc

        return result if isinstance(result, str) else tool.format_result(result)

    # ------------------------------------------------------------------ #
    def execute_batch(self, calls: list[ToolCall]) -> list[tuple[ToolCall, str]]:
        """Execute calls sequentially, preserving order for message pairing."""
        return [(call, self.execute(call)) for call in calls]
