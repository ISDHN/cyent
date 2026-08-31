"""Tests for the output parser (JSON tolerance + tool_calls extraction)."""

import pytest

from cyent.core.parser import (
    parse_json_lenient,
    parse_response,
    parse_tool_arguments,
    repair_hint,
)
from cyent.core.types import ChatResult, Message, ToolCall


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('{"a": 1, "b": [1, 2,]}', {"a": 1, "b": [1, 2]}),
        ('junk before {"cmd": "ls"} junk after', {"cmd": "ls"}),
        ("{'path': 'x.py'}", {"path": "x.py"}),
        ("{\u201cpath\u201d: \u201cx.py\u201d}", {"path": "x.py"}),
    ],
)
def test_parse_json_lenient(raw, expected):
    assert parse_json_lenient(raw) == expected


def test_parse_garbage_returns_none():
    assert parse_json_lenient("totally not json") is None
    assert parse_json_lenient("") is None


def test_parse_tool_arguments_empty_is_ok():
    args, err = parse_tool_arguments("")
    assert args == {} and err is None


def test_parse_tool_arguments_invalid():
    args, err = parse_tool_arguments("{broken")
    assert args is None and err is not None


def test_parse_tool_arguments_rejects_array():
    args, err = parse_tool_arguments("[1, 2]")
    assert args is None and "object" in err


def test_parse_response_mixed_text_and_tools():
    msg = Message.assistant(
        content="Let me check.",
        tool_calls=[
            ToolCall(id="c1", name="read_file", raw_arguments='{"path": "a.txt"}'),
            ToolCall(id="c2", name="run_command", raw_arguments="{bad json"),
        ],
    )
    parsed = parse_response(ChatResult(message=msg))
    assert parsed.text == "Let me check."
    assert len(parsed.tool_calls) == 2
    assert parsed.tool_calls[0].arguments == {"path": "a.txt"}
    assert "c2" in parsed.parse_errors  # broken args flagged...
    assert parsed.tool_calls[1].id == "c2"  # ...but call kept for pairing


def test_repair_hint_mentions_json():
    assert "JSON" in repair_hint("bad")
