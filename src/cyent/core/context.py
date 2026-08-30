"""Context manager — message history, token budget, trimming/summarization.

Core invariant: every ``assistant`` message carrying ``tool_calls`` must be
followed by exactly one ``tool`` result message per call id, in order.
Trimming and summarization must never break this pairing (OpenAI returns 400
otherwise).
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

from cyent.core.types import Message

log = logging.getLogger("cyent.context")

# Rough context budget (chars//4 estimate); conservative default.
DEFAULT_TOKEN_BUDGET = 24_000
RESERVE_FOR_REPLY = 2_000


@dataclass
class ContextStats:
    messages: int = 0
    approx_tokens: int = 0
    trims: int = 0
    summaries: int = 0


class ContextManager:
    """Owns the message stack and the token budget."""

    def __init__(
        self, system_prompt: str, token_budget: int = DEFAULT_TOKEN_BUDGET
    ) -> None:
        self._system = Message.system(system_prompt)
        self._messages: list[Message] = []
        self._budget = token_budget
        self.stats = ContextStats()
        self._on_append: list[Callable[[Message], None]] = []  # observers

    # Persistence hook: session archives subscribe here; context just
    # broadcasts every appended message and stays archive-agnostic.
    def subscribe(self, observer: Callable[[Message], None]) -> None:
        """Call ``observer(message)`` after every appended message."""
        self._on_append.append(observer)

    def _notify(self, message: Message) -> None:
        for observer in self._on_append:
            try:
                observer(message)
            except Exception:  # noqa: BLE001 — persistence must not kill runs
                log.exception("context observer failed")

    # Appends (pairing-safe)
    def add_user(self, content: str) -> Message:
        msg = Message.user(content)
        self._messages.append(msg)
        self._notify(msg)
        return msg

    def add_assistant(self, message: Message) -> Message:
        if message.role != "assistant":
            raise ValueError("add_assistant expects an assistant message")
        self._messages.append(message)
        self._notify(message)
        return message

    def add_tool_result(
        self, tool_call_id: str, content: str, name: str | None = None
    ) -> Message:
        msg = Message.tool_result(tool_call_id, content, name)
        self._messages.append(msg)
        self._notify(msg)
        return msg

    # Introspection
    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    def restore(self, messages: list[Message]) -> None:
        """Replace history (no broadcast: archives already hold these)."""
        self._messages = list(messages)

    def messages_for_api(self) -> list[Message]:
        return [self._system, *self._messages]

    def approx_tokens(self) -> int:
        return sum(m.approx_tokens() for m in self.messages_for_api())

    def stats_snapshot(self) -> ContextStats:
        self.stats.messages = len(self._messages)
        self.stats.approx_tokens = self.approx_tokens()
        return self.stats

    # Budget management
    def over_budget(self) -> bool:
        return self.approx_tokens() > self._budget - RESERVE_FOR_REPLY

    def trim_if_needed(self) -> bool:
        """Compress when over budget: summarize first (keeps information),
        trim only as a last resort. Returns True if changed."""
        if not self.over_budget():
            return False
        self.summarize()
        if not self.over_budget():
            return True
        self.trim()
        return True

    def trim(self) -> int:
        """Drop the oldest complete user/assistant/tool rounds.

        Pairing rule: never drop an assistant-with-tool_calls without also
        dropping its tool results, and never start the kept history with a
        dangling ``tool`` message. Returns the number of messages dropped.
        """
        before = len(self._messages)
        target = self._budget - RESERVE_FOR_REPLY

        while self.approx_tokens() > target and len(self._messages) > 2:
            # find the end of the first complete "round": from the first
            # message up to (and including) the last tool result of the first
            # assistant-with-tool_calls block (or just that single message if
            # it is a plain user/assistant text message).
            drop_until = self._first_round_end()
            if drop_until <= 0:
                break
            # Safety: if dropping this round would leave nothing (or leave a
            # single message that still exceeds the budget), stop trimming and
            # let summarize() handle the compression instead.
            if len(self._messages) - drop_until < 2:
                break
            del self._messages[:drop_until]
            self.stats.trims += 1
            log.info(
                "trim: dropped %d messages, ~%d tokens left",
                drop_until,
                self.approx_tokens(),
            )

        self._ensure_no_dangling_tool()
        return before - len(self._messages)

    def _first_round_end(self) -> int:
        """Index one past the end of the first removable round."""
        for i, msg in enumerate(self._messages):
            if msg.role == "assistant" and msg.has_tool_calls:
                # consume the assistant + all immediately following tool msgs
                j = i + 1
                while j < len(self._messages) and self._messages[j].role == "tool":
                    j += 1
                return j
            if msg.role in ("user", "assistant"):
                return i + 1
        return 0

    def _ensure_no_dangling_tool(self) -> None:
        """Drop leading tool messages that lost their assistant partner."""
        while self._messages and self._messages[0].role == "tool":
            del self._messages[0]

    # ------------------------------------------------------------------ #
    def summarize(self) -> bool:
        """Replace the oldest half of the history with a compact summary.

        The summary is a synthetic user message ("[context summary] ...")
        followed by the most recent messages (kept pairing-complete).
        Repeats until under budget or nothing left to compress.
        """
        changed = False
        while len(self._messages) >= 4 and self.over_budget():
            keep_from = self._safe_split_point(len(self._messages) // 2)
            if keep_from <= 1:
                # Nothing safely splittable (the front is one tool block);
                # re-summarizing would not shrink anything — stop and let
                # trim() handle the rest.
                break
            old = self._messages[:keep_from]
            kept = self._messages[keep_from:]

            summary = self._summarize_messages(old)
            summary_msg = Message.user(
                f"[context summary of earlier conversation]\n{summary}"
            )
            self._messages = [summary_msg, *kept]
            self.stats.summaries += 1
            changed = True
            log.info(
                "summarize: %d messages -> 1 summary + %d kept (~%d tokens)",
                len(old),
                len(kept),
                self.approx_tokens(),
            )
        self._ensure_no_dangling_tool()
        return changed

    def _safe_split_point(self, index: int) -> int:
        """Move ``index`` so that kept messages start a valid pairing block."""
        index = max(1, min(index, len(self._messages)))
        # If the message at `index` is a tool result, walk back to include its
        # assistant partner in the summarized part (tool msgs must not lead).
        while index < len(self._messages) and self._messages[index].role == "tool":
            index -= 1
        return max(1, index)

    def _summarize_messages(self, messages: list[Message]) -> str:
        """Extractive summary: role-tagged first lines + tool call names."""
        lines: list[str] = []
        for m in messages:
            text = (m.text() or "").strip().replace("\n", " ")
            if not text:
                continue
            tag = m.role
            if m.role == "assistant" and m.has_tool_calls:
                calls = ", ".join(tc.name for tc in m.tool_calls or [])
                lines.append(f"[{tag}] (tool calls: {calls}) {text[:120]}")
            else:
                lines.append(f"[{tag}] {text[:160]}")
        body = "\n".join(lines[-40:])  # cap the summary size
        return body or "(no content)"
