"""Error types and retry/backoff helpers (exponential backoff with jitter)."""


import logging
import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

log = logging.getLogger("cyent.errors")


class CyentError(Exception):
    """Base class for all Cyent errors."""


class ConfigError(CyentError):
    """Invalid or missing configuration (.env)."""


class LLMError(CyentError):
    """Model API failure after retries."""


class RateLimitError(LLMError):
    """429 / quota exhausted."""


class AuthError(LLMError):
    """401 / 403 — do not retry, surface config hint."""


class ContextTooLongError(LLMError):
    """Context window exceeded — caller should trim/summarize and retry."""


class ToolError(CyentError):
    """Tool execution failure (converted to observation text, never fatal)."""


class ToolTimeoutError(ToolError):
    """Tool exceeded its time budget."""


class ToolValidationError(ToolError):
    """Tool arguments failed validation."""


# --------------------------------------------------------------------------- #
# Retry with exponential backoff + jitter
# --------------------------------------------------------------------------- #
def is_retryable(exc: Exception) -> bool:
    """Classify OpenAI SDK errors: retry transient failures only."""
    import openai

    if isinstance(exc, AuthError):
        return False
    if isinstance(exc, (RateLimitError, ContextTooLongError)):
        return True
    # Generic LLMError from our own client wrapper: treat as transient
    # (the client already classified hard failures into specific types).
    if isinstance(exc, LLMError):
        return True
    if isinstance(exc, openai.RateLimitError):
        return True
    if isinstance(exc, openai.APIConnectionError | openai.APITimeoutError):
        return True
    if isinstance(exc, openai.InternalServerError):  # 5xx
        return True
    if isinstance(exc, openai.BadRequestError):
        text = str(exc).lower()
        # context-length style 400s are "retryable" after trimming
        return (
            "context length" in text
            or "maximum context" in text
            or "too many tokens" in text
        )
    return False


def with_retries(
    fn: Callable[[], T],
    *,
    max_attempts: int = 4,
    base_delay: float = 1.0,
    max_delay: float = 20.0,
    on_retry: Callable[[Exception, int], None] | None = None,
) -> T:
    """Run ``fn`` with exponential backoff + jitter on retryable errors.

    Non-retryable errors (auth, bad request other than context-length)
    propagate immediately.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — classified below
            if not is_retryable(exc) or attempt == max_attempts:
                raise
            last_exc = exc
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay *= 0.5 + random.random()  # jitter
            log.warning(
                "Attempt %d/%d failed (%s); retrying in %.1fs",
                attempt,
                max_attempts,
                exc.__class__.__name__,
                delay,
            )
            if on_retry:
                on_retry(exc, attempt)
            time.sleep(delay)
    raise last_exc  # pragma: no cover
