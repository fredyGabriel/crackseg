from __future__ import annotations

import logging


def setup_logging(
    verbose: bool | None = None, log_level: str | None = None
) -> logging.Logger:
    """Setup logging configuration for scripts.

    Args:
        verbose: When True, sets DEBUG; when False, INFO. If None, uses log_level.
        log_level: Explicit level name (e.g., "DEBUG", "INFO"). Overrides verbose when provided.

    Returns:
        The root logger configured with the chosen level and a standard format.
    """
    if log_level:
        level = getattr(logging, log_level.upper(), logging.INFO)
    elif verbose is not None:
        level = logging.DEBUG if verbose else logging.INFO
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger()
