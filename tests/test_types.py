"""Tests for core data structures (OpenAI protocol alignment)."""

from cyent.core.types import ChatResult, Message, ToolCall, ToolSchema, Usage


def test_message_roundtrip_user():
    m = Message.user("hello")
    data = m.to_openai()
    assert data == {"role": "user", "content": "hello"}
    assert Message.from_openai(data).content == "hello"


def test_message_assistant_with_tool_calls_roundtrip():
    m = Message.assistant(
        content="checking",
        tool_calls=[
            ToolCall(id="c1", name="read_file", raw_arguments='{"path": "a.txt"}')
        ],
    )
    data = m.to_openai()
    assert data["role"] == "assistant"
    assert data["tool_calls"][0]["function"]["name"] == "read_file"
    assert data["tool_calls"][0]["type"] == "function"
    back = Message.from_openai(data)
    assert back.tool_calls is not None
    assert back.tool_calls[0].id == "c1"
    assert back.tool_calls[0].raw_arguments == '{"path": "a.txt"}'


def test_message_tool_result_roundtrip():
    m = Message.tool_result("c1", "file content", "read_file")
    data = m.to_openai()
    assert data["role"] == "tool"
    assert data["tool_call_id"] == "c1"
    assert data["content"] == "file content"


def test_assistant_content_none_becomes_none_not_missing():
    m = Message.assistant(
        content=None, tool_calls=[ToolCall(id="c", name="t", raw_arguments="{}")]
    )
    data = m.to_openai()
    assert "content" in data  # key must exist for the API


def test_tool_schema_to_openai():
    s = ToolSchema(
        name="t", description="d", parameters={"type": "object", "properties": {}}
    )
    wire = s.to_openai()
    assert wire["type"] == "function"
    assert wire["function"]["name"] == "t"


def test_usage_from_openai_object():
    class FakeUsage:
        prompt_tokens = 10
        completion_tokens = 5
        total_tokens = 15

    u = Usage.from_openai(FakeUsage())
    assert (u.prompt_tokens, u.completion_tokens, u.total_tokens) == (10, 5, 15)


def test_usage_from_none():
    assert Usage.from_openai(None).total_tokens == 0


def test_approx_tokens_positive():
    assert Message.user("x" * 400).approx_tokens() > 50
