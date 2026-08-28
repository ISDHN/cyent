"""Logging layer: leveled logging to a rotating file only (never the terminal).

The REPL owns the terminal; log output there would corrupt the streaming
rendering. All records go to ``logs/cyent.log`` with secret redaction.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from cyent.config.env import Settings

_CONFIGURED = False


class RedactFilter(logging.Filter):
    """Mask registered secrets in every emitted record."""

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self._settings = settings

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        try:
            msg = record.getMessage()
            record.msg = self._settings.redact(msg)
            record.args = None
        except Exception:  # never break logging
            pass
        return True


def setup_logging(settings: Settings) -> logging.Logger:
    """Configure the root 'cyent' logger once; return the package logger."""
    global _CONFIGURED
    logger = logging.getLogger("cyent")

    if _CONFIGURED:
        return logger

    level = getattr(logging, settings.log_level, logging.INFO)
    logger.setLevel(level)
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    redact_filter = RedactFilter(settings)

    try:
        log_dir: Path = settings.log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / "cyent.log",
            maxBytes=2 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)  # file always keeps full detail
        file_handler.setFormatter(fmt)
        file_handler.addFilter(redact_filter)
        logger.addHandler(file_handler)
    except OSError:
        # No file handler available; keep logs in memory (NullHandler) so the
        # terminal is never polluted.
        logger.addHandler(logging.NullHandler())
        logger.warning("Could not create log file handler; file logging disabled.")

    _CONFIGURED = True
    logger.debug(
        "Logging initialized (level=%s, dir=%s)", settings.log_level, settings.log_dir
    )
    return logger


def get_logger(name: str = "cyent") -> logging.Logger:
    return logging.getLogger(name)
