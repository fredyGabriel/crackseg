"""Training loop component.

Handles the main training loop and epoch-level training operations.
"""

import time
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from crackseg.training.batch_processing import train_step
from crackseg.utils.checkpointing.helpers import (
    CheckpointConfig,
    CheckpointContext,
    handle_epoch_checkpointing,
)
from crackseg.utils.logging.training import safe_log
from crackseg.utils.training.amp_utils import (
    amp_autocast,
    optimizer_step_with_accumulation,
)
from crackseg.utils.training.scheduler_helper import step_scheduler_helper


class TrainingLoop:
    """Handles the main training loop and epoch operations."""

    def __init__(self) -> None:
        """Initialize the training loop component."""
        pass

    def train(self, trainer_instance: Any) -> dict[str, float]:
        """Runs the full training loop starting from self.start_epoch."""
        start_msg = (
            f"Starting training from epoch {trainer_instance.start_epoch}..."
        )
        safe_log(trainer_instance.internal_logger, "info", start_msg)
        trainer_instance.callback_handler.on_train_begin()
        start_time = time.time()
        final_val_results = {}

        for epoch in range(
            trainer_instance.start_epoch, int(trainer_instance.epochs) + 1
        ):
            trainer_instance.callback_handler.on_epoch_begin(epoch)
            epoch_start_time = time.time()
            safe_log(
                trainer_instance.internal_logger,
                "info",
                f"--- Epoch {epoch}/{trainer_instance.epochs} ---",
            )

            train_loss = self._train_epoch(trainer_instance, epoch)
            val_results = trainer_instance.validate(epoch)
            final_val_results = val_results  # Keep track of last results

            # --- Handle LR Scheduling ---
            current_lr = None
            if trainer_instance.scheduler:
                current_lr = self._step_scheduler(
                    trainer_instance, val_results
                )
                if current_lr is not None and trainer_instance.logger_instance:
                    trainer_instance.logger_instance.log_scalar(
                        tag="lr", value=current_lr, step=epoch
                    )

            # Log epoch summary using MetricsManager
            train_metrics = {"loss": train_loss}
            trainer_instance.metrics_manager.log_epoch_summary(
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics={
                    k.replace("val_", ""): v for k, v in val_results.items()
                },
                learning_rate=current_lr,
            )

            # --- Checkpointing Logic ---
            checkpoint_context = CheckpointContext(
                epoch=epoch,
                val_results=val_results,
                monitor_metric=trainer_instance.monitor_metric,
                monitor_mode=trainer_instance.monitor_mode,
                best_metric_value=trainer_instance.best_metric_value,
                experiment_config=self._safe_config_to_dict(
                    trainer_instance.cfg
                ),
            )
            checkpoint_config = CheckpointConfig(
                checkpoint_dir=str(trainer_instance.checkpoint_dir),
                logger=trainer_instance.internal_logger,
                keep_last_n=trainer_instance.cfg.get(
                    "checkpoints.keep_last_n", 1
                ),
                last_filename=trainer_instance.cfg.get(
                    "checkpoints.last_filename", "checkpoint_last.pth"
                ),
                best_filename=trainer_instance.best_filename,
            )

            trainer_instance.best_metric_value = handle_epoch_checkpointing(
                context=checkpoint_context,
                config=checkpoint_config,
                model=trainer_instance.model,
                optimizer=trainer_instance.optimizer,
            )

            # Save configuration alongside checkpoints
            self._save_epoch_configuration(trainer_instance, epoch)

            # --- Early Stopping Check ---
            if trainer_instance.early_stopper is not None and getattr(
                trainer_instance.early_stopper, "enabled", False
            ):
                current_metric = val_results.get(
                    trainer_instance.monitor_metric
                )
                if trainer_instance.early_stopper.should_stop(current_metric):
                    safe_log(
                        trainer_instance.internal_logger,
                        "info",
                        "Early stopping triggered.",
                    )
                    break  # Exit training loop

            trainer_instance.callback_handler.on_epoch_end(
                epoch, logs=val_results
            )
            epoch_duration = time.time() - epoch_start_time
            safe_log(
                trainer_instance.internal_logger,
                "info",
                f"Epoch {epoch} finished in {epoch_duration:.2f}s",
            )

        total_time = time.time() - start_time
        safe_log(
            trainer_instance.internal_logger,
            "info",
            f"Training finished in {total_time:.2f}s",
        )

        # Export final metrics summary
        summary_path = (
            trainer_instance.metrics_manager.export_metrics_summary()
        )
        trainer_instance.callback_handler.on_train_end()
        safe_log(
            trainer_instance.internal_logger,
            "info",
            f"Final metrics summary exported to: {summary_path}",
        )

        return final_val_results

    def _train_epoch(self, trainer_instance: Any, epoch: int) -> float:
        """Runs a single training epoch."""
        trainer_instance.model.train()
        total_loss = 0.0
        num_batches = len(trainer_instance.train_loader)
        log_interval = trainer_instance.cfg.get("log_interval_batches", 0)
        trainer_instance.optimizer.zero_grad()

        for batch_idx, batch in enumerate(trainer_instance.train_loader):
            trainer_instance.callback_handler.on_batch_begin(batch_idx)
            with amp_autocast(trainer_instance.use_amp):
                metrics = train_step(
                    model=trainer_instance.model,
                    batch=batch,
                    loss_fn=trainer_instance.loss_fn,
                    optimizer=trainer_instance.optimizer,
                    device=trainer_instance.device,
                    metrics_dict=trainer_instance.metrics_dict,
                )
                batch_loss = metrics["loss"]

            # Convert loss to tensor if needed
            if not isinstance(batch_loss, torch.Tensor):
                batch_loss = torch.tensor(batch_loss)
            total_loss += batch_loss.item()

            optimizer_step_with_accumulation(
                optimizer=trainer_instance.optimizer,
                scaler=trainer_instance.scaler,
                loss=batch_loss,
                grad_accum_steps=trainer_instance.grad_accum_steps,
                batch_idx=batch_idx,
                use_amp=trainer_instance.use_amp,
            )

            trainer_instance.callback_handler.on_batch_end(
                batch_idx, logs={"loss": batch_loss.item()}
            )

            if (
                trainer_instance.logger_instance
                and log_interval > 0
                and (batch_idx + 1) % log_interval == 0
            ):
                global_step = (epoch - 1) * num_batches + batch_idx + 1
                trainer_instance.logger_instance.log_scalar(
                    tag="train_batch/batch_loss",
                    value=batch_loss.item(),
                    step=global_step,
                )

        avg_loss = total_loss / num_batches
        log_msg = f"Epoch {epoch} | Average Training Loss: {avg_loss:.4f}"
        safe_log(trainer_instance.internal_logger, "info", log_msg)

        if trainer_instance.logger_instance:
            trainer_instance.logger_instance.log_scalar(
                tag="train/epoch_loss", value=avg_loss, step=epoch
            )

        return avg_loss

    def _step_scheduler(
        self, trainer_instance: Any, metrics: dict[str, float] | None = None
    ) -> float | None:
        """Steps the learning rate scheduler using a helper utility."""
        return step_scheduler_helper(
            scheduler=trainer_instance.scheduler,
            optimizer=trainer_instance.optimizer,
            monitor_metric=trainer_instance.monitor_metric,
            metrics=metrics,
            logger=trainer_instance.internal_logger,
        )

    def _save_epoch_configuration(
        self, trainer_instance: Any, epoch: int
    ) -> None:
        """Save configuration alongside checkpoints."""
        if hasattr(trainer_instance, "config_storage"):
            try:
                experiment_id = getattr(
                    trainer_instance.experiment_manager,
                    "experiment_id",
                    "default_experiment",
                )

                # Save configuration with epoch information
                config_filename = f"config_epoch_{epoch:04d}"
                trainer_instance.config_storage.save_configuration(
                    config=trainer_instance.full_cfg,
                    experiment_id=experiment_id,
                    config_name=config_filename,
                    format_type="yaml",
                )

                # Note: val_results is not stored on trainer_instance, so we skip best config saving
                # The best model detection is handled by the checkpointing system

            except Exception as e:
                safe_log(
                    trainer_instance.internal_logger,
                    "warning",
                    f"Failed to save configuration for epoch {epoch}: {e}",
                )

    def _safe_config_to_dict(self, cfg: DictConfig) -> dict[str, Any] | None:
        """Safely convert OmegaConf to dict, ensuring type safety."""
        try:
            result = OmegaConf.to_container(cfg, resolve=True)
            return result if isinstance(result, dict) else None
        except Exception:
            return None

    def _check_if_best(
        self, trainer_instance: Any, metrics: dict[str, float]
    ) -> bool:
        """Checks if the current model is the best one based on the monitored metric."""
        if not trainer_instance.save_best_enabled:
            return False

        current_metric_value = metrics.get(trainer_instance.monitor_metric)
        if current_metric_value is None:
            return False

        if trainer_instance.monitor_mode == "min":
            return current_metric_value < trainer_instance.best_metric_value
        return current_metric_value > trainer_instance.best_metric_value
