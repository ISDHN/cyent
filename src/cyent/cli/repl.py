"""REPL — interactive loop, slash commands, event rendering, Ctrl+C handling.

The CLI only consumes EngineEvents published by the engine; it never touches
engine internals (interaction/engine decoupling, per the design doc).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from cyent.config.env import Settings
from cyent.core.context import ContextManager
from cyent.core.engine import Engine, EngineConfig, EngineEvent, EventType, StopReason
from cyent.llm.client import LLMClient
from cyent.log.logger import setup_logging
from cyent.tools.command_tools import RunCommandTool
from cyent.tools.executor import ToolExecutor
from cyent.tools.file_tools import build_file_tools
from cyent.tools.info_tools import build_info_tools
from cyent.utils.errors import ConfigError

log = logging.getLogger("cyent.cli")

BANNER = """\
Cyent — minimal coding agent (OpenAI-compatible)
Type a task and press Enter. Slash commands: /help /model /clear /tools /stats /quit
"""

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

HELP = """\
Slash commands:
  /help            show this help
  /model [NAME]    show or switch the model
  /clear           clear conversation context
  /tools           list available tools
  /stats           show context/token statistics
  /quit            exit Cyent (Ctrl+C twice also works)
Anything else is sent to the agent as a task.
"""

SYSTEM_PROMPT_TEMPLATE = """\
# Role

You are Cyent, a pragmatic coding agent operating directly inside the user's
project. You turn requests into working code by investigating first, editing
precisely, and verifying with real commands. You are autonomous within the
workspace: prefer acting and verifying over asking, but stop and ask when a
task is ambiguous, destructive, or outside the workspace.

# Environment

- Workspace root: {workdir} — all file tools are confined here; paths outside
  are denied.
- Platform: {platform}. Shell commands run via the system shell; mind
  platform differences (path separators, quoting, command names).
- Today's date matters for anything time-sensitive; do not assume dates.

# Tools

- read_file: read text files (optionally a 1-based line range). Output is
  line-numbered and truncated for large files.
- write_file: create or overwrite a file wholesale. Creates parent dirs.
- edit_file: replace the FIRST unique occurrence of old_text with new_text.
  old_text must match exactly (including whitespace/indentation) and appear
  exactly once; otherwise the edit is rejected. Read the file first and copy
  the exact snippet — never guess its content.
- list_dir / project_tree: explore directory structure (ignore dirs are
  skipped). Start here when unfamiliar with the repo.
- search_text: plain-text or regex search with file:line output. Prefer it
  over reading many files; use the glob filter to narrow file types.
- run_command: run shell commands (builds, tests, git, package managers).
  Output is truncated; a timeout (default 30s, max 120s) kills the whole
  process tree. Long-running servers will time out — start them only when
  the user asks, and say so.
- pwd / env_info: workspace location, OS, Python version, env vars.

Rules of engagement:
1. Investigate before you change: for non-trivial tasks, first understand the
   relevant code (project_tree / search_text / read_file), then act.
2. Make minimal, surgical edits. Match the file's existing style, formatting,
   and language. Never reorder or reformat unrelated code, never remove
   comments or existing behavior as a side effect.
3. Verify your changes: after editing, run the relevant build/tests/linters
   via run_command. Fix what breaks before declaring success. If you cannot
   verify, say so explicitly instead of claiming it works.
4. Tool arguments must be a single strict JSON object: double quotes, no
   trailing commas, no code fences, no comments.
5. Tool failures are data, not disasters: read the error, adjust the
   approach, retry differently. Never repeat an identical failing call more
   than twice; if stuck, summarize what you learned and report.
6. Batch independent reads/searches in one round when possible; keep rounds
   few and purposeful.

# Conventions

- Do what has been asked; nothing more, nothing less. Complete the current
  task fully before moving on. Do not create files proactively "for later".
- Only modify what the task requires. If you notice an unrelated bug, mention
  it in your final answer instead of fixing it unasked.
- Preserve the user's unfinished work: if a file contains incomplete edits,
  integrate around them rather than overwriting.
- Follow existing project conventions: package manager (check for
  pyproject.toml / package.json / Cargo.toml ...), test framework, code
  style. When adding dependencies, use the project's existing manager.
- Security: never commit, print, or exfiltrate secrets (.env contents, API
  keys, tokens). Never run destructive commands (rm -rf on wide paths,
  force-pushes, dropping data) without explicit user instruction.

# Communication

- Match the user's language: reply in Chinese when they write Chinese,
  English when they write English.
- Be concise and factual. Lead with the outcome; add detail only when it
  aids understanding. No filler, no apologies, no restating the task.
- Reference code as `path:line` so the user can jump to it.
- Final answer structure for coding tasks: what changed (files + brief
  why), how it was verified (commands + results), and any caveats or
  follow-ups. If you did nothing, say what you found instead.
- Never fabricate tool output, file contents, or command results. If
  information is missing, gather it with tools or say you don't know.
"""


class Repl:
    """Terminal REPL driving the engine."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.log = setup_logging(settings)

        problems = settings.validate()
        if problems:
            raise ConfigError("\n".join(problems))

        self.client = LLMClient(settings)
        self.context = ContextManager(system_prompt=self._build_system_prompt())
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
            EngineConfig(),  # no iteration cap; termination via events
        )
        self._streaming = False  # True while assistant text is mid-stream
        self._thinking = False  # True while reasoning text is mid-stream

    # ------------------------------------------------------------------ #
    def _build_system_prompt(self) -> str:
        import platform

        return SYSTEM_PROMPT_TEMPLATE.format(
            workdir=self.settings.workdir,
            platform=f"{platform.system()} {platform.release()}",
        )

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self) -> int:
        print(BANNER)
        print(f"model: {self.settings.model} | endpoint: {self.settings.base_url}")
        print(f"workspace: {self.settings.workdir}\n")

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
        self._streaming = False  # True while assistant text is being printed
        self._thinking = False  # True while reasoning text is being printed
        try:
            for event in self.engine.run(user_input):
                self._render(event)
        except KeyboardInterrupt:
            self.engine.request_interrupt()
            print("\n(interrupted — back to prompt)")

    def _break_stream(self) -> None:
        """End an in-progress streamed text block with a newline."""
        if self._streaming:
            print()
            self._streaming = False

    def _break_thinking(self) -> None:
        """End an in-progress thinking block (reset color, newline)."""
        if self._thinking:
            print(ANSI_RESET)
            self._thinking = False

    def _render(self, event: EngineEvent) -> None:
        match event.type:
            case EventType.THINKING_DELTA:
                # Reasoning content: streamed in light gray. If normal text
                # starts afterwards, the color is reset by _break_thinking.
                if not self._thinking:
                    self._break_stream()
                    print(ANSI_THINKING, end="")
                    self._thinking = True
                print(event.text, end="", flush=True)
            case EventType.TEXT_DELTA:
                # Live token streaming: print deltas inline, no newline.
                self._break_thinking()
                print(event.text, end="", flush=True)
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
                print(f"  {ANSI_CYAN}[tool] {event.tool_name}({shown}){ANSI_RESET}")
            case EventType.TOOL_RESULT:
                result = event.tool_result
                if len(result) > 400:
                    result = result[:397] + "..."
                # Failed tool runs (errors, non-zero exit, timeouts) are
                # rendered in red; successful ones in green.
                failed = self._is_failed_result(result)
                color = ANSI_RED if failed else ANSI_GREEN
                indented = "\n".join(
                    f"  {color}| {line}{ANSI_RESET}" for line in result.splitlines()
                )
                print(indented)
            case EventType.FINAL:
                # The final answer was already streamed token-by-token; just
                # close the block and print the run summary.
                self._break_thinking()
                self._break_stream()
                reason = event.stop_reason or StopReason.COMPLETED
                if reason != StopReason.COMPLETED:
                    print(f"  {ANSI_YELLOW}[stopped: {reason.value}]{ANSI_RESET}")
                stats = self.engine.stats
                print(
                    f"  ({stats.rounds} rounds, {stats.tool_calls} tool calls, "
                    f"~{stats.prompt_tokens + stats.completion_tokens} tokens)\n"
                )
            case EventType.INTERRUPTED:
                self._break_thinking()
                self._break_stream()
                print("(interrupted)")
            case EventType.ERROR:
                self._break_thinking()
                self._break_stream()
                print(f"{ANSI_RED}[error] {event.text}{ANSI_RESET}\n")

    @staticmethod
    def _is_failed_result(result: str) -> bool:
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

    # ------------------------------------------------------------------ #
    # Slash commands; returns True when the REPL should exit
    # ------------------------------------------------------------------ #
    def _handle_command(self, line: str) -> bool:
        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        match cmd:
            case "/help" | "/?":
                print(HELP)
            case "/quit" | "/exit" | "/q":
                return True
            case "/model":
                if arg:
                    self.settings.model = arg
                    print(f"model switched to {arg}")
                else:
                    print(f"current model: {self.settings.model}")
            case "/clear":
                self.context = ContextManager(system_prompt=self._build_system_prompt())
                self.engine.context = self.context
                print("context cleared.")
            case "/tools":
                for name in self.executor.tool_names:
                    tool = self.executor.get(name)
                    desc = (tool.description or "").splitlines()[0] if tool else ""
                    print(f"  {name:14s} {desc}")
            case "/stats":
                st = self.context.stats_snapshot()
                es = self.engine.stats
                print(
                    f"  messages: {st.messages} | ~tokens: {st.approx_tokens} "
                    f"| budget: {self.context._budget} | trims: {st.trims} | summaries: {st.summaries}\n"
                    f"  last run: rounds={es.rounds}, tool_calls={es.tool_calls}, stop={es.stop_reason}"
                )
            case _:
                print(f"unknown command {cmd!r}. Try /help")
        return False


def launch(workdir: Path | None = None) -> int:
    """Build settings and start the REPL."""
    settings = Settings.load(workdir=workdir)
    try:
        repl = Repl(settings)
    except ConfigError as exc:
        print("Configuration problem:\n", exc)
        print("\nCopy .env.example to .env and fill in your endpoint/key/model.")
        return 2
    return repl.run()
