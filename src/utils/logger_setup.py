"""Helper for setting up internal logger for Trainer and safe logging."""

import logging
from typing import Any


def setup_internal_logger(logger: Any) -> logging.Logger:
    """
    Returns a logger instance. If logger is None or a string, uses
    logging.getLogger.
    """
    if logger is None:
        return logging.getLogger("Trainer")
    if isinstance(logger, str):
        return logging.getLogger(logger)
    return logger


def safe_log(logger: Any, level: str, *args: Any, **kwargs: Any) -> None:
    """
    Calls logger.<level>(*args, **kwargs) if exists, else does nothing.
    """
    fn = getattr(logger, level, None)
    if callable(fn):
        fn(*args, **kwargs)
