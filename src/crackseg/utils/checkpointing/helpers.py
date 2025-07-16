"""Helper functions for checkpointing logic in training loops."""

from dataclasses import dataclass
from typing import Any, Protocol, cast

from torch.nn import Module
from torch.optim import Optimizer

from crackseg.utils.checkpointing import CheckpointSaveConfig, save_checkpoint
from crackseg.utils.logging.training import safe_log


class SafeLogProtocol(Protocol):
    def __call__(
        self, logger: Any, level: str, *args: Any, **kwargs: Any
    ) -> None: ...


safe_log = cast(SafeLogProtocol, safe_log)


@dataclass
class CheckpointContext:
    """Encapsulates the context needed for checkpointing decisions."""

    epoch: int
    val_results: dict[str, float]
    monitor_metric: str
    monitor_mode: str
    best_metric_value: float


@dataclass
class CheckpointConfig:
    """Encapsulates configuration for saving checkpoints."""

    checkpoint_dir: str
    keep_last_n: int = 1
    last_filename: str = "checkpoint_last.pth"
    best_filename: str = "model_best.pth.tar"
    logger: Any = None


def handle_epoch_checkpointing(
    *,
    context: CheckpointContext,
    config: CheckpointConfig,
    model: Module,
    optimizer: Optimizer,
) -> float:
    """Handles checkpointing logic for an epoch, including best model check.

    Args:
        context: Contextual information for checkpointing decisions.
        config: Configuration for saving checkpoints.
        model: Model to save.
        optimizer: Optimizer to save.

    Returns:
        Updated best_metric_value (float).
    """
    # Determine metric name in val_results
    metric_name = context.monitor_metric
    all_val_metrics = all(
        k.startswith("val_") for k in context.val_results.keys()
    )
    if not metric_name.startswith("val_") and all_val_metrics:
        metric_name = f"val_{context.monitor_metric}"
    current_metric = context.val_results.get(metric_name)

    is_improvement = False
    if current_metric is not None:
        if context.monitor_mode == "min":
            is_improvement = current_metric < context.best_metric_value
        elif context.monitor_mode == "max":
            is_improvement = current_metric > context.best_metric_value

    # Save last checkpoint always
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=context.epoch,
        config=CheckpointSaveConfig(
            checkpoint_dir=config.checkpoint_dir,
            filename=config.last_filename,
            additional_data={
                "metrics": context.val_results,
                "best_metric_value": context.best_metric_value,
            },
            keep_last_n=config.keep_last_n,
        ),
    )
    if config.logger:
        safe_log(
            config.logger,
            "info",
            f"Saved last checkpoint at epoch {context.epoch}.",
        )

    # Save best checkpoint if improved
    updated_best_metric_value = context.best_metric_value
    if is_improvement and current_metric is not None:
        old_best = context.best_metric_value
        updated_best_metric_value = float(current_metric)
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=context.epoch,
            config=CheckpointSaveConfig(
                checkpoint_dir=config.checkpoint_dir,
                filename=config.best_filename,
                additional_data={
                    "metrics": context.val_results,
                    "best_metric_value": updated_best_metric_value,
                },
                keep_last_n=1,
            ),
        )
        if config.logger:
            safe_log(
                config.logger,
                "info",
                f"New best metric value: {updated_best_metric_value:.4f} "
                f"(was {old_best:.4f}). Saved best checkpoint.",
            )
    elif current_metric is None and config.logger:
        safe_log(
            config.logger,
            "warning",
            f"Monitor metric '{metric_name}' not found "
            f"in validation results. Cannot track best model.",
        )

    return updated_best_metric_value
