"""Sensitive-data redaction shared by logs, tool outputs and error messages."""

from __future__ import annotations

import re
from typing import Iterable

# Common secret shapes: OpenAI-style keys, bearer tokens, generic assignments.
_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{8,}"),
    re.compile(
        r"(?i)(api[_-]?key|token|secret|password|authorization)\s*[=:]\s*['\"]?([^\s'\"]{6,})"
    ),
)

_MASK = "***REDACTED***"


def redact(text: str, extra_secrets: Iterable[str] = ()) -> str:
    """Replace known secret shapes and explicitly registered secrets with a mask."""
    if not text:
        return text
    out = text
    for pat in _PATTERNS:
        out = pat.sub(_MASK, out)
    for secret in extra_secrets:
        if secret and len(secret) >= 6:
            out = out.replace(secret, _MASK)
    return out


def redact_mapping(data: dict, extra_secrets: Iterable[str] = ()) -> dict:
    """Redact every string value in a shallow dict (used for tool outputs)."""
    return {
        k: redact(v, extra_secrets) if isinstance(v, str) else v
        for k, v in data.items()
    }
