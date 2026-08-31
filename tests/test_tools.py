"""Tests for tools and the executor (isolation, validation, boundaries)."""

import json
from pathlib import Path

import pytest

from cyent.core.types import ToolCall
from cyent.tools.command_tools import RunCommandTool
from cyent.tools.executor import ToolExecutor
from cyent.tools.file_tools import build_file_tools
from cyent.tools.info_tools import build_info_tools


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "hello.txt").write_text("line one\nline two\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text("VALUE = 42\n", encoding="utf-8")
    return tmp_path


@pytest.fixture()
def executor(workspace: Path) -> ToolExecutor:
    return ToolExecutor(
        build_file_tools(workspace)
        + build_info_tools(workspace)
        + [RunCommandTool(workspace)]
    )


def call(executor: ToolExecutor, name: str, **args) -> str:
    return executor.execute(
        ToolCall(id="t", name=name, arguments=args, raw_arguments=json.dumps(args))
    )


# ---------------- file tools ---------------- #
def test_read_file(executor):
    out = call(executor, "read_file", path="hello.txt")
    assert "line one" in out and "line two" in out
    ranged = call(executor, "read_file", path="hello.txt", start_line=2, end_line=2)
    assert "line one" not in ranged and "line two" in ranged


def test_write_then_edit(executor):
    assert call(executor, "write_file", path="n.txt", content="v1").startswith("OK")
    assert call(
        executor, "edit_file", path="n.txt", old_text="v1", new_text="v2"
    ).startswith("OK")
    assert (executor.get("edit_file").workdir / "n.txt").read_text(
        encoding="utf-8"
    ) == "v2"


def test_edit_requires_unique_match(executor):
    call(executor, "write_file", path="dup.txt", content="aXa")
    out = call(executor, "edit_file", path="dup.txt", old_text="a", new_text="b")
    assert "appears" in out


def test_edit_missing_text(executor):
    out = call(executor, "edit_file", path="hello.txt", old_text="nope", new_text="x")
    assert "not found" in out


def test_list_dir_and_search(executor):
    assert "src/" in call(executor, "list_dir", path=".")
    out = call(executor, "search_text", pattern="VALUE", path=".")
    assert "mod.py:1" in out


def test_project_tree_and_info(executor):
    assert "hello.txt" in call(executor, "project_tree")
    assert "workspace:" in call(executor, "pwd")


# ---------------- command tool ---------------- #
def test_run_command_echo(executor):
    out = call(executor, "run_command", command="echo cyent_ok")
    assert "cyent_ok" in out and "exit_code: 0" in out


def test_run_command_timeout(executor):
    import os

    cmd = "ping -n 10 127.0.0.1" if os.name == "nt" else "sleep 5"
    out = call(executor, "run_command", command=cmd, timeout=1)
    assert "timed out" in out


# ---------------- executor isolation & validation ---------------- #
def test_unknown_tool_returns_error_not_raise(executor):
    out = call(executor, "definitely_not_a_tool")
    assert "unknown tool" in out


def test_missing_required_arg(executor):
    out = call(executor, "read_file")
    assert "missing required" in out


def test_wrong_arg_type(executor):
    out = call(executor, "read_file", path=123)
    assert "must be" in out


def test_path_escape_blocked(executor, tmp_path):
    outside = tmp_path.parent / "outside_secret.txt"
    outside.write_text("secret", encoding="utf-8")
    out = call(executor, "read_file", path=str(outside))
    assert "outside the workspace" in out


def test_output_redaction(workspace: Path):
    executor = ToolExecutor(build_file_tools(workspace), secrets=["TOPSECRET-XYZ"])
    call(executor, "write_file", path="s.txt", content="key TOPSECRET-XYZ end")
    out = call(executor, "read_file", path="s.txt")
    assert "TOPSECRET-XYZ" not in out
