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
        "--env-file",
        default=None,
        help="Path to a .env file (default: search ./.env).",
    )
    args = parser.parse_args(argv)

    workdir = Path(args.workdir).resolve() if args.workdir else Path.cwd().resolve()

    if args.print_mode:
        return _run_print_mode(args, workdir)
    from cyent.cli.repl import launch

    return launch(workdir)


def _run_print_mode(args: argparse.Namespace, workdir: Path) -> int:
    """Non-interactive single-task mode: run and print the final answer.

    All wiring (settings, session, rendering) is reused from repl.py.
    """
    from cyent.cli.repl import Session, run_single_task
    from cyent.config.env import Settings
    from cyent.utils.errors import ConfigError, CyentError

    try:
        settings = Settings.load(env_file=args.env_file, workdir=workdir)
        session = Session(settings, stream=False)
    except ConfigError as exc:
        print("Configuration problem:\n", exc, file=sys.stderr)
        print(
            "\nCopy .env.example to .env and fill in your endpoint/key/model.",
            file=sys.stderr,
        )
        return 2

    try:
        final_text, stats = run_single_task(session, args.print_mode)
    except KeyboardInterrupt:
        return 130
    except CyentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if final_text and not final_text.startswith("("):
        print(final_text)
    return int(stats.stop_reason is None or stats.stop_reason.value != "completed")


if __name__ == "__main__":
    raise SystemExit(main())
