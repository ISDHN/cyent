"""Output parser — extract and repair OpenAI-format tool_calls.

Pulls native ``tool_calls`` out of an assistant message and parses the
argument JSON, with multi-level repair (code fences, quotes, trailing
commas, outermost-fragment extraction).
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from cyent.core.types import ChatResult, ToolCall

log = logging.getLogger("cyent.parser")

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


@dataclass(slots=True)
class ParsedResponse:
    """Parsed assistant response: text + tool calls (+ parse errors)."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    parse_errors: dict[str, str] = field(default_factory=dict)  # id -> error

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


def _strip_code_fences(text: str) -> str:
    """Unwrap ```json ...``` fences."""
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def _fix_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


def _fix_smart_quotes(text: str) -> str:
    """Replace curly quotes that leak from models."""
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
    """Parse a JSON object/array, trying escalating repairs.

    Levels: direct -> strip fences -> fix quotes/commas -> extract
    outermost fragment -> single-to-double quotes. None when all fail.
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
        return None, "Arguments must be a JSON object, not an array."
    return obj, None


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
            # keep the call (empty args) so pairing stays intact
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
    """Observation asking the model to re-emit valid JSON arguments."""
    return (
        "ERROR: your tool call arguments could not be parsed. "
        f"Parser said: {error_text}. "
        "Please call the tool again with arguments as a single valid JSON object "
        "(double quotes, no trailing commas, no code fences)."
    )
