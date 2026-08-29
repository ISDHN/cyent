"""CLI entry point: ``cyent`` command."""

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
    args = parser.parse_args(argv)

    workdir = Path(args.workdir).resolve() if args.workdir else Path.cwd().resolve()

    from cyent.config.env import Settings
    from cyent.log.logger import init_logging

    Settings.load(workdir=workdir)
    init_logging()

    if args.print_mode:
        return _run_print_mode(args.print_mode)
    from cyent.cli.repl import launch

    return launch()


def _run_print_mode(task: str) -> int:
    """Non-interactive single-task mode: run and print the final answer.

    All wiring (session, rendering) is reused from repl.py; configuration
    comes from the Settings singleton.
    """
    from cyent.cli.repl import Session, run_single_task
    from cyent.utils.errors import ConfigError, CyentError

    try:
        session = Session()
    except ConfigError as exc:
        print("Configuration problem:\n", exc, file=sys.stderr)
        print(
            "\nCopy .env.example to .env and fill in your endpoint/key/model.",
            file=sys.stderr,
        )
        return 2

    try:
        _, stats = run_single_task(session, task)
    except KeyboardInterrupt:
        return 130
    except CyentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return int(stats.stop_reason is None or stats.stop_reason.value != "completed")


if __name__ == "__main__":
    raise SystemExit(main())
