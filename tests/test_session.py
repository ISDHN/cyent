"""Tests for session persistence (M8): archive format, pairing validation,
redaction, resume, and CLI integration pieces."""

import json
from pathlib import Path

import pytest

from cyent.config.env import Settings
from cyent.core.context import ContextManager
from cyent.core.session import SessionStore, _valid_prefix
from cyent.core.types import Message, ToolCall


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Install a Settings singleton scoped to tmp_path for SessionStore."""
    Settings._instance = Settings(workdir=tmp_path)
    yield Settings._instance
    Settings._instance = None


# --------------------------------------------------------------------------- #
# _valid_prefix pairing validation
# --------------------------------------------------------------------------- #
def _assistant_with_calls(*ids: str) -> Message:
    return Message.assistant(
        tool_calls=[ToolCall(id=i, name="t", raw_arguments="{}") for i in ids]
    )


def test_valid_prefix_keeps_paired_history():
    msgs = [
        Message.user("hi"),
        _assistant_with_calls("a"),
        Message.tool_result("a", "ok"),
        Message.user("again"),
    ]
    assert _valid_prefix(msgs) == msgs


def test_valid_prefix_cuts_dangling_tail_block():
    msgs = [
        Message.user("hi"),
        _assistant_with_calls("a", "b"),
        Message.tool_result("a", "ok"),
    ]
    assert _valid_prefix(msgs) == [Message.user("hi")]


def test_valid_prefix_cuts_orphan_tool_message():
    msgs = [Message.user("hi"), Message.tool_result("ghost", "x")]
    assert _valid_prefix(msgs) == [Message.user("hi")]


def test_valid_prefix_cuts_out_of_order_results():
    msgs = [
        Message.user("hi"),
        _assistant_with_calls("a", "b"),
        Message.tool_result("b", "ok"),  # wrong order: 'a' pending
        Message.tool_result("a", "ok"),
    ]
    assert _valid_prefix(msgs) == [Message.user("hi")]


def test_valid_prefix_rejects_missing_ids():
    msgs = [Message.user("hi"), _assistant_with_calls(""), Message.tool_result("", "x")]
    assert _valid_prefix(msgs) == [Message.user("hi")]


# --------------------------------------------------------------------------- #
# SessionStore round-trip
# --------------------------------------------------------------------------- #
def test_store_roundtrip(settings: Settings, tmp_path: Path):
    store = SessionStore()
    sid = store.start(model="m1")
    store.append(Message.user("hello"))
    store.append(_assistant_with_calls("c1"))
    store.append(Message.tool_result("c1", "observation"))

    loaded = SessionStore().load(sid)
    assert [m.role for m in loaded] == ["user", "assistant", "tool"]
    assert loaded[0].content == "hello"
    assert loaded[1].tool_calls[0].id == "c1"
    assert loaded[2].content == "observation"


def test_empty_session_leaves_no_file(settings: Settings, tmp_path: Path):
    """start() without any append must not create anything on disk."""
    store = SessionStore()
    store.start(model="m1")
    store.flush_meta(model="m2")  # no-op on an empty session
    assert not (tmp_path / ".cyent").exists()
    assert SessionStore().list_sessions() == []


def test_store_redacts_secrets(settings: Settings, tmp_path: Path):
    store = SessionStore()
    Settings.get().register_secret("sk-super-secret")
    sid = store.start(model="m1")
    store.append(Message.user("my key is sk-super-secret ok"))

    raw = (tmp_path / ".cyent" / "sessions" / f"{sid}.jsonl").read_text("utf-8")
    assert "sk-super-secret" not in raw
    assert "***REDACTED***" in raw


def test_store_load_truncates_corrupt_tail(settings: Settings, tmp_path: Path):
    store = SessionStore()
    sid = store.start(model="m1")
    store.append(Message.user("one"))
    store.append(Message.user("two"))

    path = tmp_path / ".cyent" / "sessions" / f"{sid}.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"type": "message", "role": "user", "content": "thr')  # torn line

    loaded = SessionStore().load(sid)
    assert [m.content for m in loaded] == ["one", "two"]


def test_store_load_missing_session(settings: Settings, tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        SessionStore().load("nope")


def test_store_list_and_latest(settings: Settings, tmp_path: Path):
    s1 = SessionStore()
    id1 = s1.start(model="m1")
    s1.append(Message.user("a"))
    s2 = SessionStore()
    id2 = s2.start(model="m2")
    s2.append(Message.user("b"))

    infos = SessionStore().list_sessions()
    assert {i.id for i in infos} == {id1, id2}
    assert all(i.messages == 1 for i in infos)
    assert SessionStore().latest().id == id2  # newest mtime first


def test_store_flush_meta_updates_model(settings: Settings, tmp_path: Path):
    store = SessionStore()
    sid = store.start(model="m1")
    store.append(Message.user("x"))
    store.flush_meta(model="m2")

    infos = SessionStore().list_sessions()
    info = next(i for i in infos if i.id == sid)
    assert info.model == "m2"
    assert info.messages == 1  # meta rewrite must not lose message lines


def test_store_adopt_then_append(settings: Settings, tmp_path: Path):
    first = SessionStore()
    sid = first.start(model="m1")
    first.append(Message.user("one"))

    second = SessionStore()
    second.adopt(sid)
    second.append(Message.user("two"))

    loaded = SessionStore().load(sid)
    assert [m.content for m in loaded] == ["one", "two"]


def test_store_append_without_start_raises(settings: Settings, tmp_path: Path):
    with pytest.raises(RuntimeError):
        SessionStore().append(Message.user("x"))


# --------------------------------------------------------------------------- #
# ContextManager observer + restore
# --------------------------------------------------------------------------- #
def test_context_observer_sees_every_append():
    ctx = ContextManager(system_prompt="s")
    seen: list[Message] = []
    ctx.subscribe(seen.append)

    ctx.add_user("hi")
    ctx.add_assistant(Message.assistant("yo"))
    ctx.add_tool_result("c1", "ok")

    assert [m.role for m in seen] == ["user", "assistant", "tool"]


def test_context_observer_failure_isolated():
    ctx = ContextManager(system_prompt="s")

    def boom(msg: Message) -> None:
        raise RuntimeError("observer exploded")

    ctx.subscribe(boom)
    ctx.add_user("still works")  # must not raise
    assert len(ctx.messages) == 1


def test_context_restore_does_not_broadcast():
    ctx = ContextManager(system_prompt="s")
    seen: list[Message] = []
    ctx.subscribe(seen.append)
    ctx.restore([Message.user("old"), Message.assistant("news")])
    assert seen == []  # restored history must not be re-persisted
    assert len(ctx.messages) == 2


# --------------------------------------------------------------------------- #
# Message archive serialization
# --------------------------------------------------------------------------- #
def test_message_archive_roundtrip():
    msg = Message.assistant(
        content=None,
        tool_calls=[ToolCall(id="c1", name="read_file", raw_arguments='{"path":"x"}')],
    )
    data = msg.to_archive()
    assert data["type"] == "message"
    assert data["tool_calls"][0]["id"] == "c1"

    back = Message.from_archive(data)
    assert back.role == "assistant"
    assert back.tool_calls[0].id == "c1"
    assert back.tool_calls[0].raw_arguments == '{"path":"x"}'


def test_archive_meta_header_is_first_line(settings: Settings, tmp_path: Path):
    store = SessionStore()
    sid = store.start(model="m1")
    store.append(Message.user("first"))  # materializes the file
    path = tmp_path / ".cyent" / "sessions" / f"{sid}.jsonl"
    first = json.loads(path.read_text("utf-8").splitlines()[0])
    assert first["type"] == "meta"
    assert first["version"] == 1
    assert first["model"] == "m1"
    assert first["id"] == sid
