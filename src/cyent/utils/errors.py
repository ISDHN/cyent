"""Error types and retry/backoff helpers (exponential backoff with jitter).

Every Cyent error carries its own retry policy as a class attribute
(``retryable``), so callers need no isinstance-based classifier. OpenAI SDK
exceptions are wrapped into Cyent errors at the client boundary
(``wrap_openai_error``); the rest of the codebase only sees Cyent types.
"""

import logging
import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

log = logging.getLogger("cyent.errors")


class CyentError(Exception):
    """Base class for all Cyent errors."""

    retryable: bool = False


class ConfigError(CyentError):
    """Invalid or missing configuration (.env)."""


class LLMError(CyentError):
    """Model API failure. Subclasses set their own retry policy."""


class RateLimitError(LLMError):
    """429 / quota exhausted — transient, retry with backoff."""

    retryable = True


class AuthError(LLMError):
    """401 / 403 — do not retry, surface config hint."""

    retryable = False


class ContextTooLongError(LLMError):
    """Context window exceeded — caller should trim/summarize and retry."""

    retryable = True


class APIConnectionError(LLMError):
    """Network / timeout / 5xx endpoint failure — transient, retry with backoff."""

    retryable = True


class BadRequestError(LLMError):
    """400 other than context-length — deterministic, do not retry."""

    retryable = False


class ToolError(CyentError):
    """Tool execution failure (converted to observation text, never fatal)."""


class ToolValidationError(ToolError):
    """Tool arguments failed validation."""


# OpenAI SDK error wrapping (the only place that imports openai exceptions)
def wrap_openai_error(exc: Exception) -> CyentError:
    """Map an OpenAI SDK exception onto the matching Cyent error type.
    Non-OpenAI exceptions pass through unchanged."""
    from openai import (
        APIConnectionError as OpenAIConnectionError,
        APITimeoutError as OpenAITimeoutError,
        AuthenticationError as OpenAIAuthError,
        BadRequestError as OpenAIBadRequestError,
        InternalServerError as OpenAIInternalServerError,
        PermissionDeniedError as OpenAIPermissionDeniedError,
        RateLimitError as OpenAIRateLimitError,
    )

    if isinstance(exc, CyentError):
        return exc

    def _ctx_length(e: Exception) -> bool:
        text = str(e).lower()
        return (
            "context length" in text
            or "maximum context" in text
            or "too many tokens" in text
        )

    match exc:
        case OpenAIRateLimitError():
            return RateLimitError(f"Rate limited / quota exceeded: {exc}")
        case OpenAIAuthError() | OpenAIPermissionDeniedError():
            return AuthError(
                f"Authentication failed ({exc}). Check OPENAI_API_KEY / OPENAI_BASE_URL in .env."
            )
        case OpenAIBadRequestError() if _ctx_length(exc):
            return ContextTooLongError(str(exc))
        case OpenAIBadRequestError():
            return BadRequestError(f"Bad request: {exc}")
        case (
            OpenAIConnectionError() | OpenAITimeoutError() | OpenAIInternalServerError()
        ):
            return APIConnectionError(f"Endpoint/connection failure: {exc}")
        case _:
            return LLMError(f"API error: {exc}")


# Retry with exponential backoff + jitter
def with_retries(
    fn: Callable[[], T],
    *,
    max_attempts: int = 4,
    base_delay: float = 1.0,
    max_delay: float = 20.0,
    on_retry: Callable[[Exception, int], None] | None = None,
) -> T:
    """Run ``fn`` with exponential backoff + jitter on retryable errors."""
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except CyentError as exc:
            if not exc.retryable or attempt == max_attempts:
                raise
            last_exc = exc
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay *= 0.5 + random.random()  # jitter
            log.warning(
                f"Attempt {attempt}/{max_attempts} failed ({exc.__class__.__name__}); retrying in {delay:.1f}s"
            )
            if on_retry:
                on_retry(exc, attempt)
            time.sleep(delay)
    raise last_exc  # pragma: no cover
