"""Structured logging setup (Phase 7): console + rotating file handler."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def configure_logging() -> logging.Logger:
    logger = logging.getLogger("rag_assistant")
    if logger.handlers:  # avoid duplicate handlers if this ever runs more than once
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(LOG_DIR / "app.log", maxBytes=1_000_000, backupCount=3)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


logger = configure_logging()
