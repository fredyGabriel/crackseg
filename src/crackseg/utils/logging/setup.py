"""Logger setup and configuration utilities.

This module provides utilities for setting up and configuring logging instances
with standardized patterns across the project.
"""

import logging
from typing import Any


def setup_internal_logger(logger: Any) -> logging.Logger:
    """Setup and return a logger instance with fallback behavior.

    Args:
        logger: Logger instance, name string, or None

    Returns:
        Configured logger instance

    Note:
        If logger is None or a string, uses logging.getLogger.
        Otherwise returns the logger as-is.
    """
    if logger is None:
        return logging.getLogger("Trainer")
    if isinstance(logger, str):
        return logging.getLogger(logger)
    return logger


def setup_project_logger(
    name: str,
    level: str = "INFO",
    format_string: str | None = None,
    add_handler: bool = True,
) -> logging.Logger:
    """Setup a standardized project logger.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages
        add_handler: Whether to add a console handler

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Only add handler if none exist and add_handler is True
    if add_handler and not logger.handlers:
        handler = logging.StreamHandler()

        if format_string is None:
            format_string = (
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

        formatter = logging.Formatter(format_string)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def configure_root_logger(
    level: str = "INFO", format_string: str | None = None
) -> None:
    """Configure the root logger with project standards.

    Args:
        level: Logging level for root logger
        format_string: Custom format string for log messages
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=format_string,
        force=True,  # Override existing configuration
    )
