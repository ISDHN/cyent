"""Tests for config, logging redaction, and retry/backoff."""

import logging

import pytest

from cyent.config.env import Settings
from cyent.log.logger import RedactFilter, setup_logging
from cyent.utils.errors import AuthError, LLMError, with_retries
from cyent.utils.redact import redact, redact_mapping


def test_settings_defaults():
    s = Settings(api_key="sk-test12345678", model="m1")
    assert s.masked_key().startswith("sk-t")
    assert s.validate() == []


def test_settings_missing_key_flagged():
    s = Settings(api_key="")
    assert any("OPENAI_API_KEY" in p for p in s.validate())


def test_redact_openai_key():
    out = redact("use sk-abcdefghijklmnop1234 please")
    assert "sk-abcdefghijklmnop1234" not in out
    assert "***REDACTED***" in out


def test_redact_bearer_and_kv():
    out = redact("Authorization: Bearer zzz123456789, token=qqq1234567890")
    assert "zzz123456789" not in out and "qqq1234567890" not in out


def test_redact_extra_secrets():
    assert "XYZZY-9999" not in redact("x XYZZY-9999 y", extra_secrets=["XYZZY-9999"])


def test_redact_mapping():
    out = redact_mapping({"k": "sk-abcdefghijklmnop", "n": 5})
    assert "sk-abcdefghijklmnop" not in out["k"] and out["n"] == 5


def test_redact_filter_masks_records():
    s = Settings(api_key="sk-abcdef1234567890")
    f = RedactFilter(s)
    rec = logging.LogRecord(
        "t", logging.INFO, "f", 1, "key is sk-abcdef1234567890", None, None
    )
    assert f.filter(rec) is True
    assert "sk-abcdef1234567890" not in rec.getMessage()


def test_setup_logging_idempotent():
    s = Settings(api_key="k", log_dir=None) if False else Settings(api_key="k")
    s.log_dir = s.log_dir  # keep default
    l1 = setup_logging(s)
    n_before = len(l1.handlers)
    l2 = setup_logging(s)
    assert len(l2.handlers) == n_before  # no duplicate handlers


# ---- retry ---- #
def test_retry_success_after_failures(monkeypatch):
    import cyent.utils.errors as errmod

    monkeypatch.setattr(errmod.time, "sleep", lambda *_: None)
    state = {"n": 0}

    def flaky():
        state["n"] += 1
        if state["n"] < 3:
            raise LLMError("transient")
        return "ok"

    assert with_retries(flaky, max_attempts=4, base_delay=0.01) == "ok"
    assert state["n"] == 3


def test_retry_auth_not_retried():
    state = {"n": 0}

    def bad():
        state["n"] += 1
        raise AuthError("401")

    with pytest.raises(AuthError):
        with_retries(bad, max_attempts=5, base_delay=0.01)
    assert state["n"] == 1


def test_retry_exhaustion(monkeypatch):
    import cyent.utils.errors as errmod

    monkeypatch.setattr(errmod.time, "sleep", lambda *_: None)
    state = {"n": 0}

    def down():
        state["n"] += 1
        raise LLMError("down")

    with pytest.raises(LLMError):
        with_retries(down, max_attempts=3, base_delay=0.01)
    assert state["n"] == 3
