"""Tests for context management (pairing-safe trim/summarize) and the engine loop."""

from pathlib import Path

from cyent.core.context import ContextManager
from cyent.core.engine import Engine, EngineConfig, EventType, StopReason
from cyent.core.types import ChatResult, Message, ToolCall
from cyent.llm.client import ChatResult as CR
from cyent.tools.executor import ToolExecutor
from cyent.tools.info_tools import build_info_tools


def make_history(ctx: ContextManager, n: int, id_prefix: str = "c") -> None:
    for i in range(n):
        ctx.add_user(f"question {i} " + "x" * 200)
        ctx.add_assistant(
            Message.assistant(
                content=None,
                tool_calls=[
                    ToolCall(id=f"{id_prefix}{i}", name="read_file", raw_arguments="{}")
                ],
            )
        )
        ctx.add_tool_result(f"{id_prefix}{i}", "content " + "y" * 200, "read_file")
        ctx.add_assistant(Message.assistant(f"answer {i}"))


def make_heavy_history(ctx: ContextManager, n: int, id_prefix: str = "h") -> None:
    """History that actually exceeds a normal budget (~1.6k tokens/round)."""
    for i in range(n):
        ctx.add_user(f"question {i} " + "x" * 2_000)
        ctx.add_assistant(
            Message.assistant(
                content=None,
                tool_calls=[
                    ToolCall(id=f"{id_prefix}{i}", name="read_file", raw_arguments="{}")
                ],
            )
        )
        ctx.add_tool_result(f"{id_prefix}{i}", "content " + "y" * 2_000, "read_file")
        ctx.add_assistant(Message.assistant(f"answer {i} " + "z" * 2_000))


def assert_pairing_valid(messages: list[Message]) -> None:
    assert (
        messages[0].role != "tool"
    ), "history must not start with a dangling tool message"
    for i, m in enumerate(messages):
        if m.role == "assistant" and m.has_tool_calls:
            nxt = messages[i + 1] if i + 1 < len(messages) else None
            assert (
                nxt is not None and nxt.role == "tool"
            ), f"unpaired tool_calls at index {i}"
            assert nxt.tool_call_id == m.tool_calls[0].id


def test_trim_keeps_pairing():
    ctx = ContextManager(system_prompt="s", token_budget=600)
    make_history(ctx, 10)
    ctx.trim()
    assert_pairing_valid(ctx.messages)


def test_summarize_keeps_pairing_and_adds_summary():
    """Under budget: summarize is a no-op; over budget: one pass per call."""
    ctx = ContextManager(system_prompt="s", token_budget=100_000)
    make_history(ctx, 10, id_prefix="d")
    assert ctx.summarize() is False  # under budget -> nothing to do
    assert ctx.stats.summaries == 0

    heavy = ContextManager(system_prompt="s", token_budget=100_000)
    make_heavy_history(heavy, 10, id_prefix="d")
    heavy._budget = 8_000  # force over-budget without re-building
    assert heavy.summarize() is True
    msgs = heavy.messages
    assert msgs[0].role == "user" and "[context summary" in (msgs[0].content or "")
    assert_pairing_valid(msgs)


def test_trim_if_needed_prefers_summarize():
    """Over budget -> summarize first (info kept), trim only as fallback."""
    ctx = ContextManager(system_prompt="s", token_budget=8_000)
    make_heavy_history(ctx, 8)
    assert ctx.over_budget()  # precondition: really over
    changed = ctx.trim_if_needed()
    assert changed is True
    assert ctx.stats.summaries >= 1  # summarize ran first
    assert not ctx.over_budget()  # converged
    assert_pairing_valid(ctx.messages)


def test_summarize_loops_until_under_budget():
    """A single summarize pass may not suffice; it must repeat."""
    ctx = ContextManager(system_prompt="s", token_budget=8_000)
    make_heavy_history(ctx, 12)
    assert ctx.summarize() is True
    assert not ctx.over_budget()  # converged
    assert ctx.stats.summaries >= 2  # needed multiple passes
    assert_pairing_valid(ctx.messages)


def test_no_trim_when_under_budget():
    ctx = ContextManager(system_prompt="s", token_budget=100_000)
    ctx.add_user("tiny")
    assert ctx.trim_if_needed() is False


# ---------------- engine ---------------- #
class FakeClient:
    """Round 1: one tool call; round 2: final answer."""

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, tools=None, temperature=0.2, max_tokens=None):
        self.calls += 1
        if self.calls == 1:
            return CR(
                message=Message.assistant(
                    content="looking",
                    tool_calls=[
                        ToolCall(id="t1", name="pwd", arguments={}, raw_arguments="{}")
                    ],
                ),
                finish_reason="tool_calls",
            )
        return CR(message=Message.assistant("final answer"), finish_reason="stop")

    def chat_stream(self, *a, **k):
        raise NotImplementedError


class LoopClient:
    def __init__(self) -> None:
        self.n = 0

    def chat(self, messages, tools=None, temperature=0.2, max_tokens=None):
        self.n += 1
        return CR(
            message=Message.assistant(
                content=None,
                tool_calls=[
                    ToolCall(
                        id=f"x{self.n}", name="pwd", arguments={}, raw_arguments="{}"
                    )
                ],
            ),
            finish_reason="tool_calls",
        )

    def chat_stream(self, *a, **k):
        raise NotImplementedError


def make_engine(client, config: EngineConfig) -> Engine:
    ctx = ContextManager(system_prompt="test")
    executor = ToolExecutor(build_info_tools(Path.cwd()))
    return Engine(client, ctx, executor, config)


def test_engine_completes_with_tool_roundtrip():
    engine = make_engine(FakeClient(), EngineConfig(stream=False))
    events = list(engine.run("do a check"))
    kinds = [e.type for e in events]
    assert EventType.TOOL_START in kinds and EventType.TOOL_RESULT in kinds
    final = [e for e in events if e.type == EventType.FINAL][0]
    assert final.text == "final answer"
    assert engine.stats.stop_reason == StopReason.COMPLETED
    assert engine.stats.tool_calls == 1
    assert_pairing_valid(engine.context.messages)


def test_engine_interrupt():
    engine = make_engine(LoopClient(), EngineConfig(stream=False))

    # Consume events manually; interrupt as soon as the first tool starts.
    events = []
    for event in engine.run("x"):
        events.append(event)
        if event.type == EventType.TOOL_START:
            engine.request_interrupt()

    assert any(e.type == EventType.INTERRUPTED for e in events)
    assert engine.stats.stop_reason == StopReason.INTERRUPTED
    assert_pairing_valid(engine.context.messages)


def test_engine_no_progress_stops():
    class StuckClient(LoopClient):
        pass

    engine = make_engine(StuckClient(), EngineConfig(stream=False))
    events = list(engine.run("stuck"))
    # pwd always returns the same output -> no-progress detector must fire
    # (there is no iteration cap anymore, so NO_PROGRESS is the only exit)
    assert engine.stats.stop_reason == StopReason.NO_PROGRESS
