"""Tests for config, logging redaction, and retry/backoff."""

import logging

import pytest

from cyent.config.env import Settings
from cyent.log.logger import RedactFilter
from cyent.utils.errors import AuthError, RateLimitError, with_retries
from cyent.utils.redact import redact


def test_settings_defaults():
    s = Settings(
        api_key="sk-test12345678",
        model="m1",
        base_url="https://api.example.com/v1",
    )
    assert s.masked_key().startswith("sk-t")
    assert s.validate() == []


def test_settings_missing_key_flagged():
    s = Settings(api_key="")
    assert any("OPENAI_API_KEY" in p for p in s.validate())


def test_redact_registered_secret_only():
    # Registry-based: nothing is masked unless explicitly registered.
    out = redact("use sk-abcdefghijklmnop1234 please")
    assert "sk-abcdefghijklmnop1234" in out  # not registered -> untouched

    out = redact(
        "use sk-abcdefghijklmnop1234 please", extra_secrets=["sk-abcdefghijklmnop1234"]
    )
    assert "sk-abcdefghijklmnop1234" not in out
    assert "***REDACTED***" in out


def test_redact_extra_secrets():
    assert "XYZZY-9999" not in redact("x XYZZY-9999 y", extra_secrets=["XYZZY-9999"])


def test_redact_short_secrets_allowed():
    # No min-length restriction: any non-empty registered value is masked.
    assert "nu" not in redact("nu is a shell", extra_secrets=["nu"])


def test_secret_registry_dedup_and_empty_rejected():
    from cyent.utils.redact import SecretRegistry

    reg = SecretRegistry()
    assert reg.register("long-secret-1") is True
    assert reg.register("long-secret-1") is True  # duplicate: still fine, no dup entry
    assert reg.secrets.count("long-secret-1") == 1
    assert reg.register("ab") is True  # short values are allowed now
    assert reg.register("") is False  # only empty is rejected
    assert reg.secrets == ["long-secret-1", "ab"]


def test_settings_register_secret():
    s = Settings(api_key="sk-abcdef1234567890")
    assert s.secrets == ["sk-abcdef1234567890"]
    assert s.redact("key is sk-abcdef1234567890 end") == "key is ***REDACTED*** end"


def test_redact_filter_masks_records():
    # RedactFilter reads the Settings singleton; install one for the test.
    Settings._instance = Settings(api_key="sk-abcdef1234567890")
    try:
        f = RedactFilter()
        rec = logging.LogRecord(
            "t", logging.INFO, "f", 1, "key is sk-abcdef1234567890", None, None
        )
        assert f.filter(rec) is True
        assert "sk-abcdef1234567890" not in rec.getMessage()
    finally:
        Settings._instance = None


# ---- retry ---- #
def test_retry_success_after_failures(monkeypatch):
    import cyent.utils.errors as errmod

    monkeypatch.setattr(errmod.time, "sleep", lambda *_: None)
    state = {"n": 0}

    def flaky():
        state["n"] += 1
        if state["n"] < 3:
            raise RateLimitError("transient")
        return "ok"

    assert with_retries(flaky, max_attempts=4, base_delay=0.01) == "ok"
    assert state["n"] == 3


def test_retryability_is_type_property():
    from cyent.utils.errors import (
        APIConnectionError,
        AuthError,
        BadRequestError,
        ConfigError,
        ContextTooLongError,
        LLMError,
        RateLimitError,
        ToolError,
    )

    assert RateLimitError.retryable is True
    assert ContextTooLongError.retryable is True
    assert APIConnectionError.retryable is True
    assert AuthError.retryable is False
    assert BadRequestError.retryable is False
    assert LLMError.retryable is False  # base: not retried by default
    assert ConfigError.retryable is False
    assert ToolError.retryable is False


def test_wrap_openai_error():
    from openai import (
        APIConnectionError as OAIConn,
        AuthenticationError as OAIAuth,
        BadRequestError as OAIBad,
        InternalServerError as OAIInternal,
        RateLimitError as OAIRate,
    )

    from cyent.utils.errors import (
        APIConnectionError,
        AuthError,
        BadRequestError,
        ContextTooLongError,
        RateLimitError,
        wrap_openai_error,
    )

    def make(exc_cls, message, status=None):
        # Build SDK errors without httpx: pass pre-built kwargs via __new__.
        exc = exc_cls.__new__(exc_cls)
        Exception.__init__(exc, message)
        return exc

    assert isinstance(wrap_openai_error(make(OAIRate, "429")), RateLimitError)
    assert isinstance(wrap_openai_error(make(OAIAuth, "401")), AuthError)
    assert isinstance(
        wrap_openai_error(make(OAIBad, "maximum context length exceeded")),
        ContextTooLongError,
    )
    assert isinstance(wrap_openai_error(make(OAIBad, "invalid body")), BadRequestError)
    assert isinstance(
        wrap_openai_error(make(OAIConn, "conn reset")), APIConnectionError
    )
    assert isinstance(wrap_openai_error(make(OAIInternal, "500")), APIConnectionError)
    # Cyent errors pass through unchanged
    own = AuthError("already typed")
    assert wrap_openai_error(own) is own


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
        raise RateLimitError("down")

    with pytest.raises(RateLimitError):
        with_retries(down, max_attempts=3, base_delay=0.01)
    assert state["n"] == 3
