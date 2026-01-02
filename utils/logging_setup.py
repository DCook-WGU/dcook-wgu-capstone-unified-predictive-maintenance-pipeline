from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(
        name: str, 
        log_file: Path, 
        level=logging.INFO,
        overwrite_handlers: bool = False,
    ) -> logging.Logger:
    """
    Configure a named logger with both console + file handlers.

    Typical usage:
        logger = configure_logging("capstone", paths.logs / "silver.log")
        child = logging.getLogger("capstone.paths")  # inherits handlers from capstone

    Notes:
    - Intended to be called once per process/kernel.
    - Notebook-safe: prevents duplicate handlers unless overwrite_handlers=True.
    """

    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # False to prevent double logging, child will still propagate 
    logger.propagate = False

    if overwrite_handlers:
        logger.handlers.clear()

    if not logger.handlers:
        
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

        streamhandler = logging.StreamHandler()
        streamhandler.setFormatter(formatter)

        filehandler = logging.FileHandler(log_file, encoding="utf-8")
        filehandler.setFormatter(formatter)

        logger.addHandler(streamhandler)
        logger.addHandler(filehandler)

    return logger


# https://docs.python.org/3/howto/logging.html
# https://docs.python.org/3/library/logging.handlers.html
# https://stackoverflow.com/questions/7173033/duplicate-log-output-when-using-python-logging-module