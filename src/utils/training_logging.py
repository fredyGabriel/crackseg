"""Helpers for training logging and metric formatting."""

from typing import Any

from src.utils.logger_setup import safe_log


def format_metrics(metrics: dict[str, float]) -> str:
    """Formats a dictionary of metrics into a string."""
    formatted: list[str] = []
    for name, value in metrics.items():
        capitalized_name = name[0].upper() + name[1:]
        formatted.append(f"{capitalized_name}: {value:.4f}")
    return " | ".join(formatted)


def log_validation_results(
    logger: Any, epoch: int, metrics: dict[str, float]
) -> None:
    """Logs validation results for an epoch using the provided logger."""
    metrics_str = format_metrics(metrics)
    safe_log(
        logger, "info", f"Epoch {epoch} | Validation Results | {metrics_str}"
    )
