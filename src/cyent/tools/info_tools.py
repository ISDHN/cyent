"""Info tools: current working directory, environment info, project tree."""

import os
import platform
import sys
from pathlib import Path
from typing import Any

from cyent.tools.base import BaseTool
from cyent.tools.file_tools import WorkspaceBoundTool, _truncate


class PwdTool(WorkspaceBoundTool):
    name = "pwd"
    description = "Return the current workspace directory and the OS platform."
    parameters: dict[str, Any] = {"type": "object", "properties": {}, "required": []}

    def run(self, **_: Any) -> str:
        return f"workspace: {self.workdir}\nplatform: {platform.system()} {platform.release()}"


class EnvInfoTool(BaseTool):
    name = "env_info"
    description = (
        "Return environment information: OS, Python version, shell, and selected "
        "environment variables (values are redacted by the executor)."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "names": {
                "type": "array",
                "items": {"type": "string"},
                "description": 'Optional list of env var names to read (e.g. ["PATH"]).',
            },
        },
        "required": [],
    }

    SECRET_HINTS = ("key", "token", "secret", "password", "auth")

    def run(self, names: list[str] | None = None, **_: Any) -> str:
        lines = [
            f"os: {platform.system()} {platform.release()} ({platform.machine()})",
            f"python: {sys.version.split()[0]}",
            f"shell: {os.environ.get('SHELL') or os.environ.get('ComSpec', '?')}",
        ]
        if names:
            for name in names[:20]:
                value = os.environ.get(name, "<unset>")
                if any(h in name.lower() for h in self.SECRET_HINTS):
                    value = "<redacted>"
                lines.append(f"env {name}={value}")
        return _truncate("\n".join(lines))


class ProjectTreeTool(WorkspaceBoundTool):
    name = "project_tree"
    description = (
        "Show a compact tree of the project (directories and files), skipping "
        "common ignore directories. Good first step to explore an unfamiliar repo."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Root of the tree (default: workspace root).",
            },
            "max_entries": {
                "type": "integer",
                "description": "Max entries to show (default 200).",
            },
        },
        "required": [],
    }

    IGNORE_DIRS = {
        "__pycache__",
        ".git",
        ".venv",
        "node_modules",
        ".pytest_cache",
        "logs",
        ".ruff_cache",
        ".idea",
        ".cyent",
    }

    def run(self, path: str = ".", max_entries: int = 200, **_: Any) -> str:
        root = self.resolve_path(path or ".")
        if not root.is_dir():
            return f"ERROR: not a directory: {self.display_path(root)}"
        max_entries = max(10, min(int(max_entries or 200), 1000))
        lines: list[str] = [f"{self.display_path(root) or '.'}/"]

        def walk(d: Path, prefix: str) -> None:
            if len(lines) >= max_entries:
                lines.append(f"{prefix}... [truncated]")
                return
            try:
                children = sorted(
                    d.iterdir(), key=lambda c: (c.is_file(), c.name.lower())
                )
            except OSError:
                return
            for child in children:
                if len(lines) >= max_entries:
                    lines.append(f"{prefix}... [truncated]")
                    return
                if child.is_dir():
                    if child.name in self.IGNORE_DIRS:
                        continue
                    lines.append(f"{prefix}{child.name}/")
                    walk(child, prefix + "  ")
                else:
                    lines.append(f"{prefix}{child.name}")

        walk(root, "  ")
        return _truncate("\n".join(lines))


def build_info_tools(workdir: Path) -> list[BaseTool]:
    """Construct all info tools bound to a workspace root."""
    return [PwdTool(workdir), EnvInfoTool(), ProjectTreeTool(workdir)]
