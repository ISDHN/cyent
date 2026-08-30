"""REPL — interactive loop, slash commands, event rendering, Ctrl+C handling.

The CLI only consumes EngineEvents published by the engine; it never touches
engine internals (interaction/engine decoupling, per the design doc).

Slash commands live in ``commands.py``; the system prompt lives in
``prompts.py``.
"""

import logging
import sys
from typing import TextIO

from cyent.cli.commands import CommandRegistry, default_commands
from cyent.cli.prompts import build_system_prompt
from cyent.config.env import Settings
from cyent.core.context import ContextManager
from cyent.core.engine import (
    Engine,
    EngineConfig,
    EngineEvent,
    EngineStats,
    EventType,
    StopReason,
)
from cyent.llm.client import LLMClient
from cyent.tools.command_tools import RunCommandTool
from cyent.tools.executor import ToolExecutor
from cyent.tools.file_tools import build_file_tools
from cyent.tools.info_tools import build_info_tools
from cyent.utils.errors import ConfigError

log = logging.getLogger("cyent.cli")

# Banner is assembled at print time so it can use the ANSI constants below.
BANNER_TITLE = "Cyent — minimal coding agent (OpenAI-compatible)"
BANNER_HINT = (
    "Type a task and press Enter. Slash commands: "
    "/help /model /clear /tools /stats /quit"
)

# ANSI color controls (ASCII escape sequences).
ANSI_RESET = "\x1b[0m"
ANSI_DIM = "\x1b[2m"  # dim / faint
# Near-white gray for thinking text: 256-color palette #252 (one step lighter
# than #250). Barely distinguishable from normal text on some themes — that
# is intentional: thinking should be visible but recede into the background.
ANSI_THINKING = "\x1b[38;5;252m"
ANSI_CYAN = "\x1b[36m"  # tool call name + args
ANSI_GREEN = "\x1b[32m"  # successful tool result
ANSI_RED = "\x1b[31m"  # failed tool result (ERROR / non-zero exit / timeout)
ANSI_YELLOW = "\x1b[33m"  # warnings (stopped reasons)


class Session:
    """Assembled agent stack: client + context + executor + engine.

    Configuration comes from the Settings singleton (Settings.get()); the
    same wiring serves the interactive REPL and single-task mode.
    """

    def __init__(self, *, stream: bool = True) -> None:
        settings = Settings.get()
        self.log = logging.getLogger("cyent")

        problems = settings.validate()
        if problems:
            raise ConfigError("\n".join(problems))

        self.client = LLMClient()
        self.context = ContextManager(system_prompt=build_system_prompt())
        self.executor = ToolExecutor(
            build_file_tools(settings.workdir)
            + build_info_tools(settings.workdir)
            + [RunCommandTool(settings.workdir)],
            secrets=settings.secrets,
        )
        self.engine = Engine(
            self.client,
            self.context,
            self.executor,
            EngineConfig(stream=stream),  # no iteration cap; termination via events
        )


class EventRenderer:
    """Renders EngineEvents to a stream (REPL: stdout; print mode: split)."""

    def __init__(self, engine: Engine, *, out: TextIO | None = None) -> None:
        self.engine = engine
        self._out = out if out is not None else sys.stdout
        self._streaming = False  # True while assistant text is mid-stream
        self._thinking = False  # True while reasoning text is mid-stream

    # ------------------------------------------------------------------ #
    def _w(self, text: str = "", *, end: str = "\n", flush: bool = False) -> None:
        print(text, file=self._out, end=end, flush=flush)

    def _break_stream(self) -> None:
        """End an in-progress streamed text block with a newline."""
        if self._streaming:
            self._w(flush=True)
            self._streaming = False

    def _break_thinking(self) -> None:
        """End an in-progress thinking block (reset color)."""
        if self._thinking:
            self._w(ANSI_RESET, end="")
            self._thinking = False

    # ------------------------------------------------------------------ #
    def render(self, event: EngineEvent) -> None:
        match event.type:
            case EventType.THINKING_DELTA:
                # Reasoning content: streamed in light gray. If normal text
                # starts afterwards, the color is reset by _break_thinking.
                if not self._thinking:
                    self._break_stream()
                    self._w(ANSI_THINKING, end="")
                    self._thinking = True
                self._w(event.text, end="", flush=True)
            case EventType.TEXT_DELTA:
                # Live token streaming: print deltas inline, no newline.
                self._break_thinking()
                self._w(event.text, end="", flush=True)
                self._streaming = True
            case EventType.ROUND_START:
                if event.round > 1:
                    self._break_thinking()
                    self._break_stream()
            case EventType.TOOL_START:
                # A tool call interrupts any streamed text; close the block.
                self._break_thinking()
                self._break_stream()
                args = event.tool_args
                shown = args if len(args) <= 120 else args[:117] + "..."
                self._w(f"  {ANSI_CYAN}[tool] {event.tool_name}({shown}){ANSI_RESET}")
            case EventType.TOOL_RESULT:
                result = event.tool_result
                if len(result) > 400:
                    result = result[:397] + "..."
                # Failed tool runs (errors, non-zero exit, timeouts) are
                # rendered in red; successful ones in green.
                failed = self.is_failed_result(result)
                color = ANSI_RED if failed else ANSI_GREEN
                indented = "\n".join(
                    f"  {color}| {line}{ANSI_RESET}" for line in result.splitlines()
                )
                self._w(indented)
            case EventType.FINAL:
                # The final answer was already streamed token-by-token; just
                # close the block and print the run summary.
                self._break_thinking()
                self._break_stream()
                reason = event.stop_reason or StopReason.COMPLETED
                if reason != StopReason.COMPLETED:
                    self._w(f"  {ANSI_YELLOW}[stopped: {reason.value}]{ANSI_RESET}")
                stats = self.engine.stats
                self._w(
                    f"  ({stats.rounds} rounds, {stats.tool_calls} tool calls, "
                    f"~{stats.prompt_tokens + stats.completion_tokens} tokens)\n"
                )
            case EventType.INTERRUPTED:
                self._break_thinking()
                self._break_stream()
                self._w("(interrupted)")
            case EventType.ERROR:
                self._break_thinking()
                self._break_stream()
                self._w(f"{ANSI_RED}[error] {event.text}{ANSI_RESET}\n")

    @staticmethod
    def is_failed_result(result: str) -> bool:
        """Heuristic: did this tool observation report a failure?"""
        head = result[:200].lstrip().lower()
        return (
            head.startswith("error")
            or head.startswith("exit_code: 1")
            or head.startswith("exit_code: -")
            or "timed out" in head
            or "exit_code: 1" in result[:120]
            or "exit_code: 2" in result[:120]
            or "exit_code: -1" in result[:120]
        )


def run_single_task(session: Session, task: str) -> tuple[str, EngineStats]:
    """Run one task non-interactively and render events.

    Assistant text/thinking stream to stdout; tool activity goes to stderr so
    the final answer can be piped cleanly. Returns (final_text, stats).
    """
    renderer_out = EventRenderer(session.engine)
    renderer_err = EventRenderer(session.engine, out=sys.stderr)
    final_text = ""
    try:
        for event in session.engine.run(task):
            # Text/thinking -> stdout; tool activity -> stderr.
            if event.type in (EventType.TEXT_DELTA, EventType.THINKING_DELTA):
                renderer_out.render(event)
            else:
                renderer_err.render(event)
            if event.type == EventType.FINAL:
                final_text = event.text
    except KeyboardInterrupt:
        session.engine.request_interrupt()
        print("\n(interrupted)", file=sys.stderr)
        raise
    return final_text, session.engine.stats


class Repl:
    """Terminal REPL driving the engine."""

    def __init__(self, *, commands: CommandRegistry | None = None) -> None:
        self.session = Session()
        self.settings = Settings.get()
        self.log = self.session.log
        self.client = self.session.client
        self.context = self.session.context
        self.executor = self.session.executor
        self.engine = self.session.engine
        self.renderer = EventRenderer(self.engine)
        # Slash commands: default set, or a custom registry for extension.
        self.commands = commands or CommandRegistry(default_commands())

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self) -> int:
        print(f"{ANSI_CYAN}{BANNER_TITLE}{ANSI_RESET}")
        print(f"{ANSI_THINKING}{BANNER_HINT}{ANSI_RESET}")
        print(
            f"{ANSI_CYAN}model:{ANSI_RESET} {self.settings.model} "
            f"{ANSI_CYAN}| endpoint:{ANSI_RESET} {self.settings.base_url}"
        )
        print(f"{ANSI_CYAN}workspace:{ANSI_RESET} {self.settings.workdir}\n")

        while True:
            try:
                user_input = input(f"[{self.settings.model}] > ").strip()
            except EOFError, KeyboardInterrupt:
                print("\nbye.")
                return 0

            if not user_input:
                continue

            if user_input.startswith("/"):
                if self._handle_command(user_input):
                    return 0
                continue

            try:
                self._run_task(user_input)
            except KeyboardInterrupt:
                # Ctrl+C during a run: interrupt the current round, keep REPL alive
                self.engine.request_interrupt()
                print("\n(interrupted — back to prompt)")
            except ConfigError as exc:
                print(f"config error: {exc}")
            except Exception as exc:  # noqa: BLE001 — REPL must survive anything
                self.log.exception("unexpected REPL error")
                print(f"error: {exc.__class__.__name__}: {exc}")

    # ------------------------------------------------------------------ #
    def _run_task(self, user_input: str) -> None:
        """Consume engine events and render them (streaming)."""
        try:
            for event in self.engine.run(user_input):
                self.renderer.render(event)
        except KeyboardInterrupt:
            self.engine.request_interrupt()
            print("\n(interrupted — back to prompt)")

    # ------------------------------------------------------------------ #
    # Slash commands: dispatch via the registry (extensible)
    # ------------------------------------------------------------------ #
    def _handle_command(self, line: str) -> bool:
        """Execute a '/...' line. Returns True when the REPL should exit."""
        return self.commands.dispatch(self, line)


def launch(*, commands: CommandRegistry | None = None) -> int:
    """Start the REPL (settings singleton must already be loaded).

    ``commands``: optional custom CommandRegistry to extend/replace the
    built-in slash commands.
    """
    try:
        repl = Repl(commands=commands)
    except ConfigError as exc:
        print("Configuration problem:\n", exc)
        print("\nCopy .env.example to .env and fill in your endpoint/key/model.")
        return 2
    return repl.run()
