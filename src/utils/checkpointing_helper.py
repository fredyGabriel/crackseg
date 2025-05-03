"""Helper functions for checkpointing logic in training loops."""

from typing import Any, Dict
from torch.nn import Module
from torch.optim import Optimizer
from src.utils.checkpointing import save_checkpoint
from src.utils.logger_setup import safe_log


def handle_epoch_checkpointing(
    *,
    epoch: int,
    model: Module,
    optimizer: Optimizer,
    val_results: Dict[str, float],
    monitor_metric: str,
    monitor_mode: str,
    best_metric_value: float,
    checkpoint_dir: str,
    logger: Any = None,
    keep_last_n: int = 1,
    last_filename: str = "checkpoint_last.pth",
    best_filename: str = "model_best.pth.tar",
) -> float:
    """Handles checkpointing logic for an epoch, including best model check.

    Args:
        epoch: Current epoch number.
        model: Model to save.
        optimizer: Optimizer to save.
        val_results: Validation metrics for the epoch.
        monitor_metric: Metric to monitor for best model.
        monitor_mode: 'min' or 'max'.
        best_metric_value: Current best value of the monitored metric.
        checkpoint_dir: Directory to save checkpoints.
        logger: Logger instance (optional).
        keep_last_n: Number of last checkpoints to keep.
        last_filename: Filename for the last checkpoint.
        best_filename: Filename for the best checkpoint.

    Returns:
        Updated best_metric_value (float).
    """
    # Determine metric name in val_results
    metric_name = monitor_metric
    all_val_metrics = all(k.startswith("val_") for k in val_results.keys())
    if not metric_name.startswith("val_") and all_val_metrics:
        metric_name = f"val_{monitor_metric}"
    current_metric = val_results.get(metric_name)

    is_improvement = False
    if current_metric is not None:
        if monitor_mode == "min":
            is_improvement = current_metric < best_metric_value
        elif monitor_mode == "max":
            is_improvement = current_metric > best_metric_value

    # Save last checkpoint always
    save_checkpoint(
        epoch=epoch,
        model=model,
        optimizer=optimizer,
        additional_data={
            "metrics": val_results,
            "best_metric_value": best_metric_value,
        },
        checkpoint_dir=checkpoint_dir,
        keep_last_n=keep_last_n,
        filename=last_filename,
    )
    if logger:
        safe_log(
            logger, "info",
            f"Saved last checkpoint at epoch {epoch}."
        )

    # Save best checkpoint if improved
    if is_improvement:
        old_best = best_metric_value
        best_metric_value = current_metric
        save_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            additional_data={
                "metrics": val_results,
                "best_metric_value": best_metric_value,
            },
            checkpoint_dir=checkpoint_dir,
            keep_last_n=1,  # Only keep the best checkpoint
            filename=best_filename,
        )
        if logger:
            safe_log(
                logger, "info",
                f"New best metric value: {best_metric_value:.4f} "
                f"(was {old_best:.4f}). Saved best checkpoint."
            )
    elif current_metric is None and logger:
        safe_log(
            logger, "warning",
            f"Monitor metric '{metric_name}' not found "
            f"in validation results. Cannot track best model."
        )

    return best_metric_value
