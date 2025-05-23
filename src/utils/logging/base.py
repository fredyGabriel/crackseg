"""Base logging functionality."""

import logging
from abc import ABC, abstractmethod
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a configured logger with the given name and level.

    Args:
        name: The name for the logger, typically the module name.
        level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level.upper())
    return logger


class BaseLogger(ABC):
    """Abstract base class for metrics and training loggers."""

    @abstractmethod
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value."""
        pass

    @abstractmethod
    def log_config(self, config: dict[str, Any]) -> None:
        """Log the experiment configuration."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the logger and release resources."""
        pass


def flatten_dict(d: dict[str, Any], parent_key: str = "") -> dict[str, Any]:
    """Flatten a nested dictionary.

    Args:
        d: Dictionary to flatten
        parent_key: Parent key for nested dictionaries

    Returns:
        Flattened dictionary with concatenated keys
    """
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}/{k}" if parent_key else k
        if isinstance(v, DictConfig):
            # Convert DictConfig to a standard dict before recursive call
            container = OmegaConf.to_container(v, resolve=True)
            if isinstance(container, dict):
                items.extend(
                    flatten_dict(
                        cast(dict[str, Any], container), new_key
                    ).items()
                )
        elif isinstance(v, dict):
            items.extend(
                flatten_dict(cast(dict[str, Any], v), new_key).items()
            )
        elif isinstance(v, int | float | str | bool):
            items.append((new_key, v))
    return dict(items)


def log_metrics_dict(
    logger: BaseLogger, metrics: dict[str, float], step: int, prefix: str = ""
) -> None:
    """Logs a dictionary of scalar metrics using the provided logger.

    Args:
        logger: An instance of BaseLogger
        metrics: Dictionary where keys are metric names and values are scalars
        step: The logging step (e.g., epoch number or global step)
        prefix: Optional prefix to add to metric tags (e.g., 'train/')
    """
    # Debug info
    standard_logger = get_logger(__name__)
    standard_logger.debug(
        f"log_metrics_dict called with logger: {logger}, "
        f"metrics: {metrics}, step: {step}, prefix: {prefix}"
    )

    if not logger:
        return  # Do nothing if logger is None

    for name, value in metrics.items():
        # Only log numeric types (int, float), explicitly excluding bool
        if isinstance(value, int | float) and not isinstance(value, bool):
            tag = f"{prefix}{name}" if prefix else name
            try:
                # Ensure value is float for logger
                logger.log_scalar(tag=tag, value=float(value), step=step)
            except Exception as e:
                # Log a warning if a specific metric fails to log
                standard_logger = get_logger(__name__)
                standard_logger.warning(
                    f"Failed to log metric '{tag}' at step {step}: {e}"
                )


class NoOpLogger(BaseLogger):
    """A logger that performs no operations, used for disabling logging."""

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Does nothing."""
        pass

    def log_config(self, config: dict[str, Any]) -> None:
        """Does nothing."""
        pass

    def close(self) -> None:
        """Does nothing."""
        pass
