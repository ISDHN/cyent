"""Sensitive-data redaction shared by logs, tool outputs and error messages.

Registry-based only: secrets must be explicitly registered (via
``SecretRegistry.register`` or ``Settings.register_secret``) before they are
masked. No shape-guessing regexes — a value is redacted if and only if it was
declared sensitive.
"""

from typing import Iterable

_MASK = "***REDACTED***"


class SecretRegistry:
    """Holds the set of sensitive values to mask, with safe registration."""

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        self._secrets: list[str] = []
        for s in secrets:
            self.register(s)

    def register(self, secret: str) -> bool:
        """Register a secret value. Returns True if it was added.

        Empty values are ignored (they would match everything).
        """
        if not secret:
            return False
        if secret not in self._secrets:
            self._secrets.append(secret)
        return True

    @property
    def secrets(self) -> list[str]:
        return list(self._secrets)

    def redact(self, text: str) -> str:
        """Replace every registered secret occurrence with the mask."""
        if not text:
            return text
        for secret in self._secrets:
            if secret in text:
                text = text.replace(secret, _MASK)
        return text


# Module-level default registry, for callers that don't own a Settings.
_DEFAULT_REGISTRY = SecretRegistry()


def redact(text: str, extra_secrets: Iterable[str] = ()) -> str:
    """Mask registered secrets (default registry plus ``extra_secrets``)."""
    if not text:
        return text
    for secret in (*_DEFAULT_REGISTRY.secrets, *extra_secrets):
        if secret and secret in text:
            text = text.replace(secret, _MASK)
    return text
