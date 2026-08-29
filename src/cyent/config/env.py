"""Configuration layer: .env loading, global Settings, secret registration."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

from cyent.utils.redact import SecretRegistry


@dataclass(slots=True)
class Settings:
    """Global runtime configuration, loaded from environment / .env."""

    base_url: str = ""
    api_key: str = ""
    model: str = ""
    log_level: str = "INFO"
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    workdir: Path = field(default_factory=Path.cwd)
    _secret_registry: SecretRegistry = field(default_factory=SecretRegistry)

    def __post_init__(self) -> None:
        self.register_secret(self.api_key)

    # ------------------------------------------------------------------ #
    @classmethod
    def load(
        cls, env_file: str | Path | None = None, workdir: Path | None = None
    ) -> Settings:
        """Load .env (if present) and build the settings object."""
        load_dotenv(env_file if env_file else None)
        return cls(
            base_url=os.getenv("OPENAI_BASE_URL", "").rstrip("/"),
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model=os.getenv("OPENAI_MODEL", ""),
            log_level=os.getenv("CYENT_LOG_LEVEL", "INFO").upper(),
            log_dir=Path(os.getenv("CYENT_LOG_DIR", "logs")),
            workdir=(workdir or Path.cwd()).resolve(),
        )

    # ------------------------------------------------------------------ #
    def register_secret(self, secret: str) -> bool:
        """Register a secret so redact() masks it everywhere.

        Returns True if the value was accepted (non-empty, long enough).
        """
        return self._secret_registry.register(secret)

    @property
    def secrets(self) -> list[str]:
        return self._secret_registry.secrets

    def redact(self, text: str) -> str:
        return self._secret_registry.redact(text)

    def masked_key(self) -> str:
        """Display-safe representation of the API key."""
        if not self.api_key:
            return "(missing)"
        if len(self.api_key) <= 8:
            return "***"
        return f"{self.api_key[:4]}...{self.api_key[-4:]}"

    def validate(self) -> list[str]:
        """Return a list of configuration problems (empty = OK)."""
        problems: list[str] = []
        if not self.api_key:
            problems.append("OPENAI_API_KEY is not set — check your .env file.")
        if not self.base_url:
            problems.append("OPENAI_BASE_URL is empty.")
        if not self.model:
            problems.append("OPENAI_MODEL is empty.")
        return problems
