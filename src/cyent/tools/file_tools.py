"""File tools: read / write / edit / list directory / search.

All paths are confined to the workspace root (path whitelist) unless
``allow_outside`` is explicitly enabled.
"""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path
from typing import Any

from cyent.tools.base import BaseTool

MAX_READ_CHARS = 40_000
MAX_WRITE_CHARS = 200_000
MAX_OUTPUT_CHARS = 20_000


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text) - limit} more chars]"


class WorkspaceBoundTool(BaseTool):
    """Tool bound to a workspace root; resolves and checks every path."""

    def __init__(self, workdir: Path, allow_outside: bool = False) -> None:
        self.workdir = workdir.resolve()
        self.allow_outside = allow_outside

    # ------------------------------------------------------------------ #
    def resolve_path(self, raw: str) -> Path:
        """Resolve ``raw`` against the workdir and enforce the whitelist."""
        if not raw or not raw.strip():
            raise ValueError("path must be a non-empty string")
        p = Path(raw)
        if not p.is_absolute():
            p = self.workdir / p
        p = p.resolve()
        if not self.allow_outside:
            try:
                p.relative_to(self.workdir)
            except ValueError:
                raise ValueError(
                    f"path {raw!r} is outside the workspace ({self.workdir}); "
                    "access denied"
                ) from None
        return p

    def display_path(self, p: Path) -> str:
        try:
            return str(p.relative_to(self.workdir))
        except ValueError:
            return str(p)


class ReadFileTool(WorkspaceBoundTool):
    name = "read_file"
    description = (
        "Read a text file and return its content. Optionally read a line range "
        "(1-based, inclusive). Lines are prefixed with line numbers."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path (relative to workspace or absolute).",
            },
            "start_line": {
                "type": "integer",
                "description": "1-based start line (optional).",
            },
            "end_line": {
                "type": "integer",
                "description": "1-based end line, inclusive (optional).",
            },
        },
        "required": ["path"],
    }

    def run(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
        **_: Any,
    ) -> str:
        p = self.resolve_path(path)
        if not p.exists():
            return f"ERROR: file not found: {self.display_path(p)}"
        if p.is_dir():
            return f"ERROR: {self.display_path(p)} is a directory, not a file"
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"ERROR: cannot read {self.display_path(p)}: {exc}"

        lines = text.splitlines()
        total = len(lines)
        s = max(1, start_line or 1)
        e = min(total, end_line or total)
        if s > total:
            return f"ERROR: start_line {s} beyond file length ({total} lines)"
        selected = lines[s - 1 : e]
        numbered = "\n".join(f"{s + i:6d}\t{line}" for i, line in enumerate(selected))
        header = f"[{self.display_path(p)}] lines {s}-{e} of {total}"
        return _truncate(f"{header}\n{numbered}")


class WriteFileTool(WorkspaceBoundTool):
    name = "write_file"
    description = (
        "Create or overwrite a text file with the given content. "
        "Parent directories are created automatically. Returns a diff-style confirmation."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path (relative to workspace or absolute).",
            },
            "content": {
                "type": "string",
                "description": "Full new content of the file.",
            },
        },
        "required": ["path", "content"],
    }

    def run(self, path: str, content: str, **_: Any) -> str:
        if len(content) > MAX_WRITE_CHARS:
            return (
                f"ERROR: content too large ({len(content)} chars > {MAX_WRITE_CHARS})"
            )
        p = self.resolve_path(path)
        existed = p.exists()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        except OSError as exc:
            return f"ERROR: cannot write {self.display_path(p)}: {exc}"
        action = "overwrote" if existed else "created"
        nlines = content.count("\n") + (
            0 if content.endswith("\n") or not content else 1
        )
        return f"OK: {action} {self.display_path(p)} ({len(content)} chars, ~{nlines} lines)"


class EditFileTool(WorkspaceBoundTool):
    name = "edit_file"
    description = (
        "Replace the FIRST unique occurrence of `old_text` with `new_text` in a file. "
        "old_text must match exactly (including whitespace) and appear exactly once; "
        "otherwise the edit is rejected. Use read_file first to get exact text."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path (relative to workspace or absolute).",
            },
            "old_text": {
                "type": "string",
                "description": "Exact text to replace (must be unique in the file).",
            },
            "new_text": {"type": "string", "description": "Replacement text."},
        },
        "required": ["path", "old_text", "new_text"],
    }

    def run(self, path: str, old_text: str, new_text: str, **_: Any) -> str:
        p = self.resolve_path(path)
        if not p.exists() or p.is_dir():
            return f"ERROR: file not found: {self.display_path(p)}"
        try:
            text = p.read_text(encoding="utf-8", errors="strict")
        except (OSError, UnicodeDecodeError) as exc:
            return f"ERROR: cannot read {self.display_path(p)}: {exc}"

        count = text.count(old_text)
        if count == 0:
            return (
                "ERROR: old_text not found in file. Read the file again and copy the "
                "exact text (whitespace matters)."
            )
        if count > 1:
            return f"ERROR: old_text appears {count} times; provide a longer, unique snippet."
        try:
            new_text_full = text.replace(old_text, new_text, 1)
            p.write_text(new_text_full, encoding="utf-8")
        except OSError as exc:
            return f"ERROR: cannot write {self.display_path(p)}: {exc}"
        return f"OK: edited {self.display_path(p)} (1 replacement, {len(old_text)} -> {len(new_text)} chars)"


class ListDirTool(WorkspaceBoundTool):
    name = "list_dir"
    description = (
        "List a directory's entries (files and subdirectories) with sizes. "
        "Optionally recursive up to `depth` levels. Directories end with '/'."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path (default: workspace root).",
            },
            "depth": {
                "type": "integer",
                "description": "Recursion depth (default 1, max 4).",
            },
        },
        "required": [],
    }

    def run(self, path: str = ".", depth: int = 1, **_: Any) -> str:
        d = self.resolve_path(path or ".")
        if not d.exists():
            return f"ERROR: directory not found: {self.display_path(d)}"
        if not d.is_dir():
            return f"ERROR: {self.display_path(d)} is a file, not a directory"
        depth = max(1, min(int(depth or 1), 4))
        entries: list[str] = []

        def walk(dir_path: Path, level: int, prefix: str) -> None:
            try:
                children = sorted(
                    dir_path.iterdir(), key=lambda c: (c.is_file(), c.name.lower())
                )
            except OSError as exc:
                entries.append(f"{prefix}[error: {exc}]")
                return
            for child in children:
                if child.is_dir():
                    entries.append(f"{prefix}{child.name}/")
                    if level < depth:
                        walk(child, level + 1, prefix + "  ")
                else:
                    try:
                        size = child.stat().st_size
                    except OSError:
                        size = -1
                    entries.append(f"{prefix}{child.name} ({size} bytes)")

        walk(d, 1, "")
        return _truncate(
            f"[{self.display_path(d)}] depth={depth}\n" + "\n".join(entries)
        )


class SearchTool(WorkspaceBoundTool):
    name = "search_text"
    description = (
        "Search files under a directory for a text pattern (plain text or regex). "
        "Returns matching lines with file:line prefixes. Respects common ignore dirs."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Text or regular expression to find.",
            },
            "path": {
                "type": "string",
                "description": "Directory or file to search (default: workspace root).",
            },
            "is_regex": {
                "type": "boolean",
                "description": "Treat pattern as regex (default false).",
            },
            "glob": {
                "type": "string",
                "description": "Filename filter, e.g. '*.py' (default: all files).",
            },
            "max_results": {
                "type": "integer",
                "description": "Max matches returned (default 50).",
            },
        },
        "required": ["pattern"],
    }

    IGNORE_DIRS = {
        "__pycache__",
        ".git",
        ".venv",
        "node_modules",
        ".pytest_cache",
        "logs",
        ".ruff_cache",
    }

    def run(
        self,
        pattern: str,
        path: str = ".",
        is_regex: bool = False,
        glob: str = "*",
        max_results: int = 50,
        **_: Any,
    ) -> str:
        root = self.resolve_path(path or ".")
        if not root.exists():
            return f"ERROR: path not found: {self.display_path(root)}"
        max_results = max(1, min(int(max_results or 50), 200))

        if is_regex:
            try:
                rx = re.compile(pattern)
            except re.error as exc:
                return f"ERROR: invalid regex: {exc}"
            matcher = lambda s: rx.search(s)  # noqa: E731
        else:
            needle = pattern.lower()
            matcher = lambda s: needle in s.lower()  # noqa: E731

        files: list[Path] = []
        if root.is_file():
            files = [root]
        else:
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in self.IGNORE_DIRS]
                for fname in filenames:
                    if fnmatch.fnmatch(fname, glob):
                        files.append(Path(dirpath) / fname)

        results: list[str] = []
        truncated = False
        for f in files:
            if len(results) >= max_results:
                truncated = True
                break
            try:
                if f.stat().st_size > 2_000_000:
                    continue
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                if matcher(line):
                    results.append(
                        f"{self.display_path(f)}:{lineno}: {line.strip()[:200]}"
                    )
                    if len(results) >= max_results:
                        truncated = True
                        break

        if not results:
            return f"No matches for {pattern!r} under {self.display_path(root)}"
        suffix = f"\n... [results truncated at {max_results}]" if truncated else ""
        return _truncate("\n".join(results) + suffix)


def build_file_tools(workdir: Path, allow_outside: bool = False) -> list[BaseTool]:
    """Construct all file tools bound to a workspace root."""
    return [
        ReadFileTool(workdir, allow_outside),
        WriteFileTool(workdir, allow_outside),
        EditFileTool(workdir, allow_outside),
        ListDirTool(workdir, allow_outside),
        SearchTool(workdir, allow_outside),
    ]
