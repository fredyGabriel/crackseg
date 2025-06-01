"""Helper functions for learning rate schedulers."""

from typing import Any, cast

import torch

from src.utils.logger_setup import safe_log


def step_scheduler_helper(
    *,
    scheduler: Any | None,
    optimizer: Any | None,
    monitor_metric: str,
    metrics: dict[str, float] | None = None,
    logger: Any | None = None,
) -> float | None:
    """Steps the learning rate scheduler, handling ReduceLROnPlateau and
    others.

    Args:
        scheduler: The scheduler instance (can be None).
        optimizer: The optimizer instance (can be None).
        monitor_metric: Name of the metric to monitor (for ReduceLROnPlateau).
        metrics: Dict of metrics from validation (optional).
        logger: Logger for warnings/info (optional).

    Returns:
        The current learning rate after stepping, or None if not available.
    """
    if scheduler is None:
        return None

    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        monitor_value = None
        if metrics is not None:
            monitor_value = metrics.get(monitor_metric)
        if monitor_value is None:
            if logger:
                safe_log(
                    logger,
                    "warning",
                    f"ReduceLROnPlateau scheduler needs '{monitor_metric}' "
                    "metric for step. Skipping scheduler step.",
                )
        else:
            scheduler.step(monitor_value)  # type: ignore[reportUnknownMemberType]
    else:
        scheduler.step()

    if optimizer is not None:
        current_lr = optimizer.param_groups[0]["lr"]
        if logger:
            safe_log(
                logger,
                "info",
                f"LR Scheduler step. Current LR: {current_lr:.6f}",
            )
        return cast(float, current_lr)
    return None
