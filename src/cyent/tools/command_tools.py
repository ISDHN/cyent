"""Command tool: run local commands with timeout, workdir boundary, output caps."""


import os
import subprocess
from pathlib import Path
from typing import Any

from cyent.tools.base import BaseTool
from cyent.tools.file_tools import _truncate

DEFAULT_TIMEOUT = 30
MAX_TIMEOUT = 120
MAX_OUTPUT = 20_000

# Commands that are refused unless explicitly allowed by the user.
DANGEROUS_PATTERNS: tuple[str, ...] = (
    "rm -rf /",
    "rm -rf ~",
    "del /f /s /q c:\\",
    "format ",
    "shutdown",
    "reboot",
    "mkfs",
    "dd if=",
    "> /dev/sda",
    "reg delete",
    "remove-item -recurse -force c:\\",
)


class RunCommandTool(BaseTool):
    """Execute a shell command inside the workspace and capture its output."""

    name = "run_command"
    description = (
        "Run a shell command in the workspace directory and return stdout, stderr "
        "and the exit code. Use for builds, tests, git, etc. Output is truncated. "
        "A timeout (seconds, default 30, max 120) applies."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command line to execute.",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default 30, max 120).",
            },
            "workdir": {
                "type": "string",
                "description": "Optional subdirectory of the workspace to run in.",
            },
        },
        "required": ["command"],
    }

    def __init__(
        self,
        workdir: Path,
        *,
        allow_dangerous: bool = False,
        allow_outside: bool = False,
    ) -> None:
        self.workdir = workdir.resolve()
        self.allow_dangerous = allow_dangerous
        self.allow_outside = allow_outside

    # ------------------------------------------------------------------ #
    def _check_dangerous(self, command: str) -> str | None:
        low = command.lower()
        for pat in DANGEROUS_PATTERNS:
            if pat in low:
                return (
                    f"ERROR: command matches a dangerous pattern ({pat!r}) and was blocked. "
                    "Ask the user to enable it explicitly if it is really needed."
                )
        return None

    def _resolve_workdir(self, raw: str | None) -> Path:
        if not raw:
            return self.workdir
        p = Path(raw)
        if not p.is_absolute():
            p = self.workdir / p
        p = p.resolve()
        if not self.allow_outside:
            try:
                p.relative_to(self.workdir)
            except ValueError:
                raise ValueError(f"workdir {raw!r} is outside the workspace") from None
        return p

    def run(
        self,
        command: str,
        timeout: int | None = None,
        workdir: str | None = None,
        **_: Any,
    ) -> str:
        if not command or not command.strip():
            return "ERROR: command must be a non-empty string"
        if not self.allow_dangerous:
            blocked = self._check_dangerous(command)
            if blocked:
                return blocked

        try:
            cwd = self._resolve_workdir(workdir)
        except ValueError as exc:
            return f"ERROR: {exc}"

        timeout = min(max(int(timeout or DEFAULT_TIMEOUT), 1), MAX_TIMEOUT)

        return self._execute(command, cwd, timeout)

    def _execute(self, command: str, cwd: Path, timeout: int) -> str:
        """Run the command and kill the *whole process tree* on timeout.

        On Windows, ``subprocess.run(timeout=...)`` kills only the direct
        child but then blocks draining pipes held by grandchildren — the
        terminal appears frozen ("闪退"). We therefore use Popen +
        ``taskkill /F /T`` (Windows) or ``killpg`` (POSIX) to terminate the
        entire tree, then drain the pipes.
        """
        kwargs: dict[str, Any] = {
            "shell": True,
            "cwd": str(cwd),
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "errors": "replace",
        }
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True  # own process group for killpg

        try:
            proc = subprocess.Popen(command, **kwargs)
        except OSError as exc:
            return f"ERROR: failed to start command: {exc}"

        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._kill_tree(proc)
            stdout, stderr = proc.communicate()
            return (
                f"ERROR: command timed out after {timeout}s and the process tree "
                f"was terminated: {command!r}"
            )
        except KeyboardInterrupt:
            self._kill_tree(proc)
            proc.communicate()
            raise

        parts = [f"exit_code: {proc.returncode}"]
        if stdout:
            parts.append(f"--- stdout ---\n{_truncate(stdout, MAX_OUTPUT // 2)}")
        if stderr:
            parts.append(f"--- stderr ---\n{_truncate(stderr, MAX_OUTPUT // 2)}")
        if not stdout and not stderr:
            parts.append("(no output)")
        return "\n".join(parts)

    @staticmethod
    def _kill_tree(proc: subprocess.Popen) -> None:
        """Terminate the process tree; fall back to killing the child only."""
        if proc.poll() is not None:
            return
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
            )
        else:
            import signal

            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError, PermissionError:
                proc.kill()
