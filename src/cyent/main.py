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
    parser.add_argument(
        "-c",
        "--continue",
        dest="continue_session",
        action="store_true",
        help="Continue the most recent session.",
    )
    parser.add_argument(
        "--resume",
        metavar="ID",
        default=None,
        help="Resume a specific session (see --list-sessions).",
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="List saved sessions and exit.",
    )
    args = parser.parse_args(argv)

    from cyent.config.env import Settings
    from cyent.core.session import SessionStore
    from cyent.log.logger import init_logging

    workdir = Path(args.workdir).resolve() if args.workdir else Path.cwd().resolve()
    Settings.load(workdir=workdir)
    init_logging()

    if args.list_sessions:
        sessions = SessionStore().list_sessions()
        if not sessions:
            print("no saved sessions.")
            return 0
        print(f"{'ID':24s} {'model':16s} {'msgs':>5s}  updated (UTC)")
        for s in sessions:
            updated = s.updated_at[:19].replace("T", " ")
            print(f"{s.id:24s} {s.model:16s} {s.messages:5d}  {updated}")
        return 0

    resume_id = args.resume
    if args.continue_session and not resume_id:
        latest = SessionStore().latest()
        if latest is None:
            print("no saved session to continue; starting fresh.")
        else:
            resume_id = latest.id

    if args.print_mode:
        return _run_print_mode(args.print_mode)
    from cyent.cli.repl import launch

    return launch(resume_id=resume_id)


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
