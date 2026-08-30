"""Logging: leveled records to a rotating file only (never the terminal).

The REPL owns the terminal; log output there would corrupt the streaming
rendering. All records go to ``logs/cyent.log`` with secret redaction.
"""

import logging
from logging.handlers import RotatingFileHandler

from cyent.config.env import Settings


class RedactFilter(logging.Filter):
    """Mask registered secrets in every emitted record."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
            record.msg = Settings.get().redact(msg)
            record.args = None
        except Exception:
            pass
        return True


def init_logging() -> logging.Logger:
    """Configure the 'cyent' logger. Called exactly once from main()."""
    logger = logging.getLogger("cyent")
    settings = Settings.get()
    level = getattr(logging, settings.log_level, logging.INFO)
    logger.setLevel(level)
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        log_dir = settings.log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / "cyent.log",
            maxBytes=2 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)  # file always keeps full detail
        file_handler.setFormatter(fmt)
        file_handler.addFilter(RedactFilter())
        logger.addHandler(file_handler)
    except OSError:
        # No file handler available; never pollute the terminal.
        logger.addHandler(logging.NullHandler())
        logger.warning("Could not create log file handler; file logging disabled.")

    logger.debug(
        "Logging initialized (level=%s, dir=%s)", settings.log_level, settings.log_dir
    )
    return logger
