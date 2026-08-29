"""Configuration layer: .env loading, global Settings, secret registration."""


import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from cyent.utils.redact import redact


@dataclass(slots=True)
class Settings:
    """Global runtime configuration, loaded from environment / .env."""

    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o-mini"
    log_level: str = "INFO"
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    workdir: Path = field(default_factory=Path.cwd)
    # Secrets registered for full-chain redaction (logs + tool outputs).
    _secrets: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    @classmethod
    def load(
        cls, env_file: str | Path | None = None, workdir: Path | None = None
    ) -> Settings:
        """Load .env (if present) and build the settings object."""
        load_dotenv(env_file if env_file else None)
        s = cls(
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip(
                "/"
            ),
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            log_level=os.getenv("CYENT_LOG_LEVEL", "INFO").upper(),
            log_dir=Path(os.getenv("CYENT_LOG_DIR", "logs")),
            workdir=(workdir or Path.cwd()).resolve(),
        )
        s.register_secret(s.api_key)
        return s

    # ------------------------------------------------------------------ #
    def register_secret(self, secret: str) -> None:
        """Register a secret so redact() masks it everywhere."""
        if secret and secret not in self._secrets:
            self._secrets.append(secret)

    @property
    def secrets(self) -> list[str]:
        return list(self._secrets)

    def redact(self, text: str) -> str:
        return redact(text, self._secrets)

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
