"""Session persistence — JSONL archives of the full conversation history.

One file per session under ``<workdir>/.cyent/sessions/<id>.jsonl``. The
first line is a meta document (id / model / timestamps / format version);
every following line is one message in OpenAI wire format
(``Message.to_archive``). Message appends are flushed + fsynced per line;
meta updates rewrite the file atomically (tmp + ``os.replace``).

Loading validates the ``assistant.tool_calls`` ↔ ``tool`` pairing and
truncates at the first inconsistency, so a corrupted tail never reaches
the API (which would answer 400). Message contents are redacted with the
registered secrets before hitting the disk. The system prompt is NOT
archived: it is re-rendered from the current environment on load.
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from cyent.config.env import Settings
from cyent.core.types import Message
from cyent.utils.redact import redact

log = logging.getLogger("cyent.session")

ARCHIVE_VERSION = 1
META_TYPE = "meta"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def new_session_id() -> str:
    """``<timestamp>-<4 hex>``, e.g. ``20260830-143025-a1b2``."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{os.urandom(2).hex()}"


def _valid_prefix(messages: list[Message]) -> list[Message]:
    """Longest prefix whose ``tool_calls``/``tool`` pairing is intact.

    Rules enforced while walking:
    - inside an open tool block (assistant issued calls, results pending)
      only the matching ``tool`` result for the OLDEST pending id may follow;
    - a ``tool`` message with no open block is invalid;
    - an assistant/user message while a block is open means the block is
      dangling → cut where that block began;
    - a dangling block at the end is likewise cut at its assistant.
    """
    cut: int | None = None
    pending: list[str] = []  # outstanding tool_call ids, in order
    block_start = -1  # index of the assistant that opened `pending`

    for i, m in enumerate(messages):
        if pending:
            if m.role == "tool" and m.tool_call_id == pending[0]:
                pending.pop(0)
                if not pending:
                    block_start = -1
                continue
            cut = block_start  # dangling block: cut where it began
            break
        if m.role == "assistant" and m.has_tool_calls:
            ids = [tc.id for tc in m.tool_calls or []]
            if not all(ids):
                cut = i  # ids missing → pairing unverifiable
                break
            pending = list(ids)
            block_start = i
        elif m.role in ("user", "assistant"):
            pass  # plain message, fine
        else:  # tool / system without an open block
            cut = i
            break

    if cut is None and pending:
        cut = block_start  # dangling tail block
    return messages[:cut] if cut is not None else list(messages)


@dataclass(slots=True)
class SessionInfo:
    """Summary of one archive on disk (for listings)."""

    id: str
    model: str
    created_at: str
    updated_at: str
    messages: int
    path: Path


class SessionStore:
    """Creates, appends to, and loads session archives.

    Configuration (workdir, registered secrets) comes from the Settings
    singleton — no constructor parameters.
    """

    def __init__(self) -> None:
        settings = Settings.get()
        self._dir = settings.workdir / ".cyent" / "sessions"
        self._path: Path | None = None  # currently active archive
        self._meta: dict = {}

    # ------------------------------------------------------------------ #
    @property
    def current_id(self) -> str | None:
        return self._meta.get("id") if self._meta else None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self, model: str) -> str:
        """Arm a fresh archive; the file is created lazily on first append.

        Empty sessions (no interaction) leave nothing on disk.
        Returns the prospective id.
        """
        sid = new_session_id()
        self._path = self._dir / f"{sid}.jsonl"
        self._meta = {
            "type": META_TYPE,
            "version": ARCHIVE_VERSION,
            "id": sid,
            "model": model,
            "created_at": _now(),
            "updated_at": _now(),
        }
        log.info("session armed: %s (model=%s, lazy file)", sid, model)
        return sid

    def adopt(self, session_id: str) -> None:
        """Point the store at an existing archive (to continue appending)."""
        path = self._dir / f"{session_id}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"no session archive {session_id!r}")
        self._meta = self._read_meta(path)
        self._path = path
        log.info("session adopted: %s", session_id)

    # ------------------------------------------------------------------ #
    # Writing
    # ------------------------------------------------------------------ #
    def append(self, message: Message) -> None:
        """Append one message (redacted) to the active archive.

        The archive file is created on the first append (meta header first),
        so sessions with no interaction never touch the disk.
        """
        if self._path is None:
            raise RuntimeError("no active session archive (start/adopt first)")
        data = message.to_archive()
        if isinstance(data.get("content"), str):
            data["content"] = redact(data["content"], Settings.get().secrets)
        if not self._path.exists():
            self._dir.mkdir(parents=True, exist_ok=True)
            header = json.dumps(self._meta, ensure_ascii=False) + "\n"
        else:
            header = ""
        with open(self._path, "a", encoding="utf-8") as f:
            if header:
                f.write(header)
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def flush_meta(self, model: str | None = None) -> None:
        """Atomically rewrite the meta header (updated_at, maybe model).

        No-op when the archive was never materialized (empty session).
        """
        if self._path is None or not self._meta or not self._path.exists():
            return
        if model:
            self._meta["model"] = model
        self._meta["updated_at"] = _now()
        lines = self._path.read_text(encoding="utf-8").splitlines(keepends=True)
        tmp = self._path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(self._meta, ensure_ascii=False) + "\n")
            if len(lines) > 1:
                f.writelines(lines[1:])
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self._path)

    # ------------------------------------------------------------------ #
    # Reading
    # ------------------------------------------------------------------ #
    def load(self, session_id: str) -> list[Message]:
        """Load a session's messages; corrupt/unpairable tails are dropped."""
        path = self._dir / f"{session_id}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"no session archive {session_id!r}")

        messages: list[Message] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("session %s: corrupt line dropped", session_id)
                    break
                if not isinstance(data, dict) or data.get("type") == META_TYPE:
                    continue
                try:
                    messages.append(Message.from_archive(data))
                except Exception:
                    log.warning("session %s: unreadable message dropped", session_id)
                    break

        valid = _valid_prefix(messages)
        dropped = len(messages) - len(valid)
        if dropped:
            log.warning(
                "session %s: dropped %d unpairable message(s)", session_id, dropped
            )
        return valid

    def list_sessions(self) -> list[SessionInfo]:
        """All archives, most recently touched first."""
        if not self._dir.exists():
            return []
        out: list[SessionInfo] = []
        for path in self._dir.glob("*.jsonl"):
            try:
                out.append(self._read_info(path))
            except OSError, ValueError, json.JSONDecodeError:
                log.warning("skipping unreadable archive %s", path.name)
        out.sort(key=lambda s: s.path.stat().st_mtime_ns, reverse=True)
        return out

    def latest(self) -> SessionInfo | None:
        sessions = self.list_sessions()
        return sessions[0] if sessions else None

    # ------------------------------------------------------------------ #
    @staticmethod
    def _read_meta(path: Path) -> dict:
        with open(path, encoding="utf-8") as f:
            first = f.readline()
        data = json.loads(first) if first.strip() else {}
        if not isinstance(data, dict) or data.get("type") != META_TYPE:
            raise ValueError(f"{path.name}: missing meta header")
        return data

    def _read_info(self, path: Path) -> SessionInfo:
        meta = self._read_meta(path)
        with open(path, encoding="utf-8") as f:
            total = sum(1 for line in f if line.strip())
        return SessionInfo(
            id=meta.get("id") or path.stem,
            model=meta.get("model", "?"),
            created_at=meta.get("created_at", "?"),
            updated_at=meta.get("updated_at", "?"),
            messages=max(0, total - 1),  # first line is meta
            path=path,
        )
