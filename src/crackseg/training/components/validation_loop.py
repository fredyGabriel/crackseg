"""Validation loop component.

Handles validation operations and metric computation during training.
"""

from typing import Any

import torch

from crackseg.training.batch_processing import val_step
from crackseg.utils.logging.training import log_validation_results


class ValidationLoop:
    """Handles validation operations and metric computation."""

    def __init__(self) -> None:
        """Initialize the validation loop component."""
        pass

    def validate(self, trainer_instance: Any, epoch: int) -> dict[str, float]:
        """Runs validation over the val_loader and aggregates metrics."""
        trainer_instance.model.eval()
        total_metrics = {"loss": 0.0}
        total_metrics.update(dict.fromkeys(trainer_instance.metrics_dict, 0.0))
        num_batches = len(trainer_instance.val_loader)

        with torch.no_grad():
            for batch in trainer_instance.val_loader:
                metrics = val_step(
                    model=trainer_instance.model,
                    batch=batch,
                    loss_fn=trainer_instance.loss_fn,
                    device=trainer_instance.device,
                    metrics_dict=trainer_instance.metrics_dict,
                )
                for name, value in metrics.items():
                    # Convert tensor to float for accumulation
                    total_metrics[name] += (
                        value.item()
                        if hasattr(value, "item")
                        else float(value)
                    )

        avg_metrics = {
            name: total / num_batches for name, total in total_metrics.items()
        }
        val_metrics = {}
        for name, value in avg_metrics.items():
            if name == "loss":
                val_metrics["val_loss"] = value
            else:
                val_metrics[f"val_{name}"] = value

        # Use MetricsManager for unified logging
        trainer_instance.metrics_manager.log_training_metrics(
            epoch=epoch,
            step=0,  # Step 0 for epoch-level validation
            metrics=val_metrics,
            phase="val",
        )

        # Legacy logging for backward compatibility
        log_validation_results(
            trainer_instance.internal_logger, epoch, val_metrics
        )

        # Original logger instance logging (if available)
        if trainer_instance.logger_instance:
            for name, value in val_metrics.items():
                trainer_instance.logger_instance.log_scalar(
                    tag=f"{name}", value=value, step=epoch
                )

        return val_metrics
