"""CLI entry point: ``cyent`` command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cyent",
        description="Cyent — a minimal coding agent over any OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "-C",
        "--workdir",
        default=None,
        help="Workspace directory (default: current directory).",
    )
    parser.add_argument(
        "-p",
        "--print",
        dest="print_mode",
        metavar="TASK",
        default=None,
        help="Run a single task non-interactively and exit.",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="Path to a .env file (default: search ./.env).",
    )
    args = parser.parse_args(argv)

    workdir = Path(args.workdir).resolve() if args.workdir else Path.cwd().resolve()

    # Import lazily so --help stays fast.
    from cyent.config.env import Settings

    if args.print_mode:
        return _run_print_mode(args, workdir)
    from cyent.cli.repl import launch

    return launch(workdir)


def _run_print_mode(args: argparse.Namespace, workdir: Path) -> int:
    """Non-interactive single-task mode: run and print the final answer."""
    from cyent.config.env import Settings
    from cyent.core.context import ContextManager
    from cyent.core.engine import Engine, EngineConfig, EventType
    from cyent.llm.client import LLMClient
    from cyent.log.logger import setup_logging
    from cyent.tools.command_tools import RunCommandTool
    from cyent.tools.executor import ToolExecutor
    from cyent.tools.file_tools import build_file_tools
    from cyent.tools.info_tools import build_info_tools
    from cyent.utils.errors import ConfigError

    settings = Settings.load(env_file=args.env_file, workdir=workdir)
    setup_logging(settings)
    problems = settings.validate()
    if problems:
        print("Configuration problem:\n" + "\n".join(problems), file=sys.stderr)
        return 2

    import platform

    from cyent.cli.repl import (
        ANSI_CYAN,
        ANSI_GREEN,
        ANSI_RED,
        ANSI_RESET,
        ANSI_THINKING,
        SYSTEM_PROMPT_TEMPLATE,
    )

    context = ContextManager(
        system_prompt=SYSTEM_PROMPT_TEMPLATE.format(
            workdir=settings.workdir,
            platform=f"{platform.system()} {platform.release()}",
        )
    )
    executor = ToolExecutor(
        build_file_tools(settings.workdir)
        + build_info_tools(settings.workdir)
        + [RunCommandTool(settings.workdir)],
        secrets=settings.secrets,
    )
    engine = Engine(
        LLMClient(settings),
        context,
        executor,
        EngineConfig(stream=False),  # no iteration cap
    )

    final_text = ""
    try:
        for event in engine.run(args.print_mode):
            if event.type == EventType.TEXT_DELTA:
                print(event.text, end="", flush=True)
            elif event.type == EventType.THINKING_DELTA:
                # reasoning output in the shared light-gray thinking color
                print(f"{ANSI_THINKING}{event.text}{ANSI_RESET}", end="", flush=True)
            elif event.type == EventType.TOOL_START:
                print(
                    f"  {ANSI_CYAN}[tool] {event.tool_name}{ANSI_RESET}",
                    file=sys.stderr,
                )
            elif event.type == EventType.TOOL_RESULT:
                failed = (
                    event.tool_result.lstrip()
                    .lower()
                    .startswith(
                        ("error", "exit_code: 1", "exit_code: 2", "exit_code: -")
                    )
                    or "timed out" in event.tool_result[:120].lower()
                )
                color = ANSI_RED if failed else ANSI_GREEN
                print(f"  {color}{event.tool_result}{ANSI_RESET}", file=sys.stderr)
            elif event.type == EventType.FINAL:
                final_text = event.text
    except KeyboardInterrupt:
        engine.request_interrupt()
        print("\n(interrupted)", file=sys.stderr)
        return 130

    if final_text and not final_text.startswith("("):
        print(final_text)
    return (
        0
        if engine.stats.stop_reason is not None
        and engine.stats.stop_reason.value == "completed"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
