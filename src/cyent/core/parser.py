"""Output parser — extract and repair OpenAI-format tool_calls.

Responsibilities:
- Pull native ``tool_calls`` (id / name / arguments string) out of an
  assistant message and parse arguments into objects.
- Multi-level JSON repair: strip code fences, fix quotes/trailing commas,
  extract the outermost JSON fragment.
- Separate coexisting text content and tool calls in one assistant message.
- No Anthropic handling (``tool_use`` / ``tool_result``) by design.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from cyent.core.types import ChatResult, ToolCall

log = logging.getLogger("cyent.parser")

# How many times we allow the model to fix malformed tool arguments.
MAX_REPAIR_ATTEMPTS = 2

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


@dataclass(slots=True)
class ParsedResponse:
    """Result of parsing one assistant response."""

    text: str = ""  # plain content (may be empty)
    tool_calls: list[ToolCall] = field(default_factory=list)
    # tool_call_id -> error text, for calls whose arguments could not be parsed
    parse_errors: dict[str, str] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


# --------------------------------------------------------------------------- #
# JSON repair helpers
# --------------------------------------------------------------------------- #
def _strip_code_fences(text: str) -> str:
    """If the payload is wrapped in ```json ...```, unwrap it."""
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def _fix_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


def _fix_smart_quotes(text: str) -> str:
    """Replace curly quotes that sometimes leak from models."""
    return (
        text.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )


def _extract_outermost_json(text: str) -> str | None:
    """Extract the first balanced {...} or [...] fragment from noisy text."""
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def parse_json_lenient(raw: str) -> dict[str, Any] | list[Any] | None:
    """Try hard to parse a JSON object/array from ``raw``.

    Levels: direct parse -> strip fences -> fix quotes/commas -> extract
    outermost fragment -> single-quote to double-quote fallback.
    Returns None when every level fails.
    """
    if not raw or not raw.strip():
        return None

    candidates = [raw]
    stripped = _strip_code_fences(raw)
    if stripped != raw.strip():
        candidates.append(stripped)
    candidates.append(_fix_smart_quotes(stripped))
    candidates.append(_fix_trailing_commas(_fix_smart_quotes(stripped)))

    frag = _extract_outermost_json(stripped)
    if frag:
        candidates.append(frag)
        candidates.append(_fix_trailing_commas(_fix_smart_quotes(frag)))

    # single-quoted JSON (common model slip)
    candidates.append(stripped.replace("'", '"'))

    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError, ValueError:
            continue
        if isinstance(obj, dict | list):
            return obj
    return None


def parse_tool_arguments(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse tool-call argument JSON. Returns (args, error)."""
    if not raw or not raw.strip():
        return {}, None  # empty args are legal
    obj = parse_json_lenient(raw)
    if obj is None:
        return None, f"Arguments are not valid JSON: {raw[:200]!r}"
    if isinstance(obj, list):
        # tolerate a bare list by wrapping positionally? No — reject clearly.
        return None, "Arguments must be a JSON object, not an array."
    return obj, None


# --------------------------------------------------------------------------- #
# Response parsing
# --------------------------------------------------------------------------- #
def parse_response(result: ChatResult) -> ParsedResponse:
    """Split one ChatResult into text + parsed tool calls (+ parse errors)."""
    msg = result.message
    parsed = ParsedResponse(text=msg.content or "")

    for tc in msg.tool_calls or []:
        args, err = parse_tool_arguments(tc.raw_arguments)
        if err is not None:
            log.warning(
                "Tool call %s (%s) has invalid arguments: %s", tc.id, tc.name, err
            )
            parsed.parse_errors[tc.id] = err
            # keep the call with empty args so pairing stays intact; the
            # executor will surface the error observation to the model.
            parsed.tool_calls.append(
                ToolCall(
                    id=tc.id, name=tc.name, arguments={}, raw_arguments=tc.raw_arguments
                )
            )
        else:
            parsed.tool_calls.append(
                ToolCall(
                    id=tc.id,
                    name=tc.name,
                    arguments=args or {},
                    raw_arguments=tc.raw_arguments,
                )
            )

    if parsed.tool_calls:
        log.info(
            "Parsed %d tool call(s): %s",
            len(parsed.tool_calls),
            [tc.name for tc in parsed.tool_calls],
        )
    return parsed


def repair_hint(error_text: str) -> str:
    """Observation text asking the model to re-emit valid JSON arguments."""
    return (
        "ERROR: your tool call arguments could not be parsed. "
        f"Parser said: {error_text}. "
        "Please call the tool again with arguments as a single valid JSON object "
        "(double quotes, no trailing commas, no code fences)."
    )
