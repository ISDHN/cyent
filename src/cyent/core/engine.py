"""Engine — the ReAct agent loop (think -> act -> observe).

Calls the model; when the model returns tool_calls, executes them locally and
feeds observations back for the next round; when no tool calls are returned,
the loop finishes with the final answer.

Termination conditions (any one ends the loop):
1. max iterations reached;
2. no tool calls in the response (final text answer);
3. user interrupt (injected via ``request_interrupt``);
4. no substantive progress for several rounds (repeated/failed tool results);
5. explicit stop flag.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum

from cyent.core.context import ContextManager
from cyent.core.parser import (
    MAX_REPAIR_ATTEMPTS,
    ParsedResponse,
    parse_response,
    repair_hint,
)
from cyent.core.types import ChatResult, Message, ToolCall
from cyent.llm.client import LLMClient, StreamEvent
from cyent.tools.executor import ToolExecutor
from cyent.utils.errors import AuthError, ContextTooLongError, LLMError

log = logging.getLogger("cyent.engine")

# Rounds without progress before we degrade and summarize.
NO_PROGRESS_LIMIT = 3


class StopReason(str, Enum):
    COMPLETED = "completed"  # model gave a final text answer
    MAX_ITERATIONS = "max_iterations"
    INTERRUPTED = "interrupted"
    NO_PROGRESS = "no_progress"
    ERROR = "error"


class EventType(str, Enum):
    TEXT_DELTA = "text_delta"
    THINKING_DELTA = "thinking_delta"  # reasoning content (rendered dim)
    ROUND_START = "round_start"
    TOOL_START = "tool_start"
    TOOL_RESULT = "tool_result"
    FINAL = "final"
    ERROR = "error"
    INTERRUPTED = "interrupted"


@dataclass(slots=True)
class EngineEvent:
    """Event published by the engine; the CLI only consumes these."""

    type: EventType
    text: str = ""
    tool_name: str = ""
    tool_args: str = ""
    tool_result: str = ""
    round: int = 0
    stop_reason: StopReason | None = None


@dataclass
class EngineConfig:
    # max_iterations removed: the loop runs until a termination condition
    # (final answer / interrupt / no-progress / error) is met.
    temperature: float = 0.2
    stream: bool = True
    max_repair_attempts: int = MAX_REPAIR_ATTEMPTS


@dataclass
class EngineStats:
    rounds: int = 0
    tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    stop_reason: StopReason | None = None


class Engine:
    """The agent loop. One ``run`` = one user task, end to end."""

    def __init__(
        self,
        client: LLMClient,
        context: ContextManager,
        executor: ToolExecutor,
        config: EngineConfig | None = None,
    ) -> None:
        self.client = client
        self.context = context
        self.executor = executor
        self.config = config or EngineConfig()
        self._interrupt = threading.Event()
        self.stats = EngineStats()
        self._last_result: ChatResult | None = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def request_interrupt(self) -> None:
        """Called by the CLI (Ctrl+C handler) to abort the current run."""
        self._interrupt.set()

    def reset_interrupt(self) -> None:
        self._interrupt.clear()

    def run(self, user_input: str) -> Iterator[EngineEvent]:
        """Run one full agent turn; yields events for the CLI to render."""
        self.reset_interrupt()
        self.stats = EngineStats()
        self.context.add_user(user_input)
        log.info("=== engine run start: %r ===", user_input[:120])

        yield from self._loop()

        log.info("=== engine run end: %s ===", self.stats.stop_reason)

    # ------------------------------------------------------------------ #
    # Core loop
    # ------------------------------------------------------------------ #
    def _loop(self) -> Iterator[EngineEvent]:
        recent_results: list[str] = []
        no_progress_rounds = 0
        repair_attempts = 0
        iteration = 0

        # No iteration cap: the loop ends via final answer, user interrupt,
        # no-progress degradation, or an unrecoverable error.
        while True:
            iteration += 1
            if self._interrupt.is_set():
                self.stats.stop_reason = StopReason.INTERRUPTED
                yield EngineEvent(
                    type=EventType.INTERRUPTED, stop_reason=StopReason.INTERRUPTED
                )
                return

            self.stats.rounds = iteration
            yield EngineEvent(type=EventType.ROUND_START, round=iteration)

            # ---- think: call the model (with retry/trim handling) ------- #
            try:
                for kind, text in self._call_model():
                    if kind == "text":
                        yield EngineEvent(type=EventType.TEXT_DELTA, text=text)
                    else:
                        yield EngineEvent(type=EventType.THINKING_DELTA, text=text)
                result = self._last_result
            except ContextTooLongError:
                log.warning("Context too long; trimming and retrying")
                self.context.trim()
                try:
                    for kind, text in self._call_model():
                        event_type = (
                            EventType.TEXT_DELTA
                            if kind == "text"
                            else EventType.THINKING_DELTA
                        )
                        yield EngineEvent(type=event_type, text=text)
                    result = self._last_result
                except ContextTooLongError:
                    self.stats.stop_reason = StopReason.ERROR
                    yield EngineEvent(
                        type=EventType.ERROR,
                        text="Context still too long after trimming.",
                    )
                    return
            except AuthError as exc:
                self.stats.stop_reason = StopReason.ERROR
                yield EngineEvent(type=EventType.ERROR, text=str(exc))
                return
            except LLMError as exc:
                self.stats.stop_reason = StopReason.ERROR
                yield EngineEvent(
                    type=EventType.ERROR, text=f"Model call failed: {exc}"
                )
                return

            self.stats.prompt_tokens += result.usage.prompt_tokens
            self.stats.completion_tokens += result.usage.completion_tokens

            # ---- parse -------------------------------------------------- #
            parsed = parse_response(result)

            # ---- no tool calls => final answer -------------------------- #
            if not parsed.has_tool_calls:
                self.context.add_assistant(result.message)
                self.stats.stop_reason = StopReason.COMPLETED
                yield EngineEvent(
                    type=EventType.FINAL,
                    text=parsed.text,
                    stop_reason=StopReason.COMPLETED,
                )
                return

            # ---- act + observe: execute tools, keep pairing ------------- #
            self.context.add_assistant(result.message)  # assistant.tool_calls first...

            for tc in parsed.tool_calls:
                if self._interrupt.is_set():
                    # still must fill remaining tool results to keep pairing
                    self.context.add_tool_result(
                        tc.id, "(interrupted by user)", tc.name
                    )
                    continue

                # argument parse error -> ask the model to fix (limited)
                if tc.id in parsed.parse_errors:
                    if repair_attempts < self.config.max_repair_attempts:
                        repair_attempts += 1
                        self.context.add_tool_result(
                            tc.id, repair_hint(parsed.parse_errors[tc.id]), tc.name
                        )
                        continue
                    self.context.add_tool_result(
                        tc.id,
                        f"ERROR: unparseable arguments: {parsed.parse_errors[tc.id]}",
                        tc.name,
                    )
                    continue

                yield EngineEvent(
                    type=EventType.TOOL_START,
                    tool_name=tc.name,
                    tool_args=tc.raw_arguments,
                    round=iteration,
                )
                observation = self.executor.execute(tc)
                self.stats.tool_calls += 1
                self.context.add_tool_result(
                    tc.id, observation, tc.name
                )  # ...then tool result
                yield EngineEvent(
                    type=EventType.TOOL_RESULT,
                    tool_name=tc.name,
                    tool_result=observation,
                    round=iteration,
                )

                # ---- no-progress detection ------------------------------ #
                recent_results.append(observation[:200])
                recent_results = recent_results[-NO_PROGRESS_LIMIT:]
                if self._is_stuck(recent_results):
                    no_progress_rounds += 1
                else:
                    no_progress_rounds = 0

            if no_progress_rounds >= NO_PROGRESS_LIMIT:
                log.warning("No progress for %d rounds; degrading", no_progress_rounds)
                self.context.add_user(
                    "SYSTEM NOTE: the last several tool results repeated or failed with no progress. "
                    "Stop retrying the same approach; summarize what you have learned so far "
                    "and give your best final answer now."
                )
                self.stats.stop_reason = StopReason.NO_PROGRESS
                yield EngineEvent(
                    type=EventType.FINAL,
                    text="(stopped: no progress)",
                    stop_reason=StopReason.NO_PROGRESS,
                )
                return

            # keep context within budget after tool results were added
            self.context.trim_if_needed()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _call_model(self) -> Iterator[EngineEvent | tuple[str, str]]:
        """Call the model, streaming deltas to the CLI when enabled.

        Yields ``(kind, text)`` tuples where kind is ``"text"`` or
        ``"thinking"``. The final ChatResult is stored on
        ``self._last_result`` for the caller.
        """
        messages = self.context.messages_for_api()
        tools = self.executor.schemas()
        if self.config.stream:
            events, holder = self.client.chat_stream(
                messages,
                tools=tools,
                temperature=self.config.temperature,
            )
            for ev in events:
                if ev.kind == "text_delta" and ev.text:
                    yield ("text", ev.text)  # live token streaming
                elif ev.kind == "thinking_delta" and ev.text:
                    yield ("thinking", ev.text)  # reasoning, rendered dim
            self._last_result = holder.get()
        else:
            self._last_result = self.client.chat(
                messages, tools=tools, temperature=self.config.temperature
            )

    @staticmethod
    def _is_stuck(recent: list[str]) -> bool:
        """True when the last tool results are identical failures/repeats."""
        if len(recent) < NO_PROGRESS_LIMIT:
            return False
        return len(set(recent)) == 1
