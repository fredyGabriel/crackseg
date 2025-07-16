"""Training-specific logging utilities.

This module provides helper functions for logging training metrics, validation
results, and epoch summaries during model training.
"""

from typing import Any

import torch

from .base import get_logger
from .metrics_manager import MetricsManager

logger = get_logger(__name__)


def format_metrics(metrics: dict[str, float]) -> str:
    """Formats a dictionary of metrics into a string.

    Args:
        metrics: Dictionary of metric names and values

    Returns:
        Formatted string with capitalized metric names
    """
    formatted: list[str] = []
    for name, value in metrics.items():
        capitalized_name = name[0].upper() + name[1:]
        formatted.append(f"{capitalized_name}: {value:.4f}")
    return " | ".join(formatted)


def safe_log(
    logger_instance: Any, level: str, *args: Any, **kwargs: Any
) -> None:
    """Safely call logger method if it exists.

    Args:
        logger_instance: Logger instance to use
        level: Log level method name (e.g., 'info', 'error')
        *args: Arguments to pass to the logger method
        **kwargs: Keyword arguments to pass to the logger method
    """
    fn = getattr(logger_instance, level, None)
    if callable(fn):
        fn(*args, **kwargs)


def log_validation_results(
    logger_instance: Any, epoch: int, metrics: dict[str, float]
) -> None:
    """Logs validation results for an epoch using the provided logger.

    Args:
        logger_instance: Logger instance to use for output
        epoch: Current epoch number
        metrics: Dictionary of metric names and values
    """
    metrics_str = format_metrics(metrics)
    safe_log(
        logger_instance,
        "info",
        f"Epoch {epoch} | Validation Results | {metrics_str}",
    )


def log_training_results(
    logger_instance: Any, epoch: int, step: int, metrics: dict[str, float]
) -> None:
    """Logs training results for a specific step using the provided logger.

    Args:
        logger_instance: Logger instance to use for output
        epoch: Current epoch number
        step: Current step/batch number
        metrics: Dictionary of metric names and values
    """
    metrics_str = format_metrics(metrics)
    safe_log(
        logger_instance,
        "info",
        f"Epoch {epoch} Step {step} | Training | {metrics_str}",
    )


def create_unified_metrics_logger(
    experiment_dir: str,
    logger_instance: Any | None = None,
    config: Any | None = None,
) -> MetricsManager:
    """Create a MetricsManager instance with standardized configuration.

    Args:
        experiment_dir: Directory for experiment outputs
        logger_instance: Optional logger instance
        config: Optional configuration object

    Returns:
        Configured MetricsManager instance
    """
    return MetricsManager(
        experiment_dir=experiment_dir,
        logger=logger_instance,
        config=config,
    )


# Legacy compatibility functions
def log_epoch_metrics(
    metrics_manager: MetricsManager,
    epoch: int,
    train_metrics: dict[str, float] | None = None,
    val_metrics: dict[str, float] | None = None,
    learning_rate: float | None = None,
) -> None:
    """Legacy wrapper for epoch metrics logging using MetricsManager.

    Args:
        metrics_manager: MetricsManager instance
        epoch: Current epoch number
        train_metrics: Training metrics for the epoch
        val_metrics: Validation metrics for the epoch
        learning_rate: Current learning rate
    """
    metrics_manager.log_epoch_summary(
        epoch=epoch,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        learning_rate=learning_rate,
    )


def log_step_metrics(
    metrics_manager: MetricsManager,
    epoch: int,
    step: int,
    metrics: dict[str, float | torch.Tensor],
    phase: str = "train",
) -> None:
    """Legacy wrapper for step-level metrics logging using MetricsManager.

    Args:
        metrics_manager: MetricsManager instance
        epoch: Current epoch number
        step: Current step/batch number
        metrics: Dictionary of metric names and values (supports tensors)
        phase: Training phase ('train', 'val', 'test')
    """
    metrics_manager.log_training_metrics(
        epoch=epoch,
        step=step,
        metrics=metrics,
        phase=phase,
    )
