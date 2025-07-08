"""Trainer: orchestrates training, validation, checkpointing, and early
stopping."""

# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false, reportUnannotatedClassAttribute=false
# pyright: reportUnknownParameterType=false, reportMissingParameterType=false
# pyright: reportAttributeAccessIssue=false, reportPrivateUsage=false
# pyright: reportExplicitAny=false, reportUnusedCallResult=false
# Global suppressions for systematic issues with configuration objects,
# PyTorch types, and complex training logic

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.training.batch_processing import train_step, val_step
from src.training.config_validation import validate_trainer_config

# Import factory functions
from src.training.factory import create_lr_scheduler, create_optimizer
from src.utils import BaseLogger
from src.utils.checkpointing import (
    CheckpointConfig,
    CheckpointContext,
    handle_epoch_checkpointing,
    load_checkpoint,
    setup_checkpointing,
)
from src.utils.config.standardized_storage import (
    StandardizedConfigStorage,
    validate_configuration_completeness,
)
from src.utils.core.device import get_device
from src.utils.logging.metrics_manager import MetricsManager
from src.utils.logging.setup import setup_internal_logger
from src.utils.logging.training import log_validation_results, safe_log
from src.utils.monitoring import (
    BaseCallback,
    CallbackHandler,
    MonitoringManager,
)
from src.utils.training.amp_utils import (
    GradScaler,
    amp_autocast,
    optimizer_step_with_accumulation,
)
from src.utils.training.early_stopping import EarlyStopping
from src.utils.training.early_stopping_setup import setup_early_stopping
from src.utils.training.scheduler_helper import step_scheduler_helper

# Placeholder: Checkpointing
# from src.utils.checkpointing import save_checkpoint, load_checkpoint
# Placeholder: Early Stopping
# from src.evaluation.early_stopping import EarlyStopping


@dataclass
class TrainingComponents:
    """Encapsulates the core components required for training."""

    model: torch.nn.Module
    train_loader: DataLoader[Any]
    val_loader: DataLoader[Any]
    loss_fn: torch.nn.Module
    metrics_dict: dict[str, Any]


class Trainer:
    """Orchestrates the training and validation process."""

    def __init__(
        self,
        components: TrainingComponents,
        cfg: DictConfig,
        logger_instance: BaseLogger | None = None,
        # Early stopper can be instantiated outside and passed, or configured
        early_stopper: EarlyStopping | None = None,
        callbacks: list[BaseCallback] | None = None,
    ):
        """Initializes the Trainer."""
        self._initialize_core_attributes(components, cfg, logger_instance)
        self._parse_trainer_settings()
        self._setup_monitoring(callbacks)  # Setup monitoring
        self._setup_checkpointing_attributes()
        self._setup_device_and_model()
        self._setup_optimizer_and_scheduler()
        self._setup_mixed_precision()
        self._load_checkpoint_state()
        self._setup_early_stopping_instance(early_stopper)
        self._log_initialization_summary()

    def _initialize_core_attributes(
        self,
        components: TrainingComponents,
        cfg: DictConfig,
        logger_instance: BaseLogger | None,
    ):
        """Validates config and initializes core trainer attributes."""
        validate_trainer_config(cfg.training)
        self.full_cfg = cfg
        self.cfg = cfg.training  # Main trainer config node
        self.model = components.model
        self.train_loader = components.train_loader
        self.val_loader = components.val_loader
        self.loss_fn = components.loss_fn
        self.metrics_dict = components.metrics_dict
        self.logger_instance = logger_instance
        self.internal_logger = setup_internal_logger(logger_instance)
        self.grad_accum_steps = self.cfg.get("gradient_accumulation_steps", 1)
        self.verbose = self.cfg.get("verbose", True)
        self.start_epoch = 1  # Default start epoch

    def _setup_monitoring(self, callbacks: list[BaseCallback] | None) -> None:
        """Initializes the monitoring and callback system."""
        self.monitoring_manager = MonitoringManager()
        self.callback_handler = CallbackHandler(
            callbacks or [], self.monitoring_manager
        )

    def _parse_trainer_settings(self):
        """Parses basic trainer settings from the configuration."""
        self.epochs = self.cfg.get("epochs", 10)
        self.device_str = self.cfg.get("device", "auto")
        self.use_amp = self.cfg.get("use_amp", True)
        self.verbose = self.cfg.get("verbose", True)
        self.start_epoch = 1  # Default start epoch

    def _setup_checkpointing_attributes(self):
        """Sets up attributes related to checkpointing."""
        (checkpoint_dir_str, self.experiment_manager) = setup_checkpointing(
            self.full_cfg,  # Pass full_cfg for setup_checkpointing
            getattr(self.logger_instance, "experiment_manager", None),
            self.internal_logger,
        )
        # Convert string to Path object
        self.checkpoint_dir = Path(checkpoint_dir_str)
        self.save_freq = self.cfg.get("save_freq", 0)
        self.checkpoint_load_path = self.cfg.get("checkpoint_load_path", None)

        save_best_config = self.cfg.get("save_best", {})
        # Fallback to checkpoints.save_best if training.save_best is not
        # present
        if not save_best_config and "checkpoints" in self.cfg:
            checkpoints_cfg = self.cfg.get("checkpoints", {})
            save_best_config = checkpoints_cfg.get("save_best", {})

        self.save_best_enabled = save_best_config.get("enabled", False)
        self.monitor_metric = save_best_config.get(
            "monitor_metric", "val_loss"
        )
        self.monitor_mode = save_best_config.get("monitor_mode", "min")
        self.best_filename = save_best_config.get(
            "best_filename", "model_best.pth.tar"
        )
        self.best_metric_value: float = (
            float("inf") if self.monitor_mode == "min" else float("-inf")
        )

        # Initialize MetricsManager for unified metric logging
        self._setup_metrics_manager()

        # Initialize StandardizedConfigStorage for configuration management
        self._setup_standardized_config_storage()

    def _setup_metrics_manager(self) -> None:
        """Initialize the MetricsManager for standardized metric logging."""
        # Get experiment directory from experiment_manager or checkpoint_dir
        experiment_dir = (
            getattr(self.experiment_manager, "experiment_dir", None)
            if self.experiment_manager
            else None
        )

        # Fallback to checkpoint_dir parent if experiment_dir is not available
        if experiment_dir is None:
            experiment_dir = self.checkpoint_dir.parent

        # Use get_logger for Python logger instead of BaseLogger
        from src.utils.logging.base import get_logger

        python_logger = get_logger("trainer.metrics")

        self.metrics_manager = MetricsManager(
            experiment_dir=experiment_dir,
            logger=python_logger,
            config=self.full_cfg,
        )

        safe_log(
            self.internal_logger,
            "info",
            f"MetricsManager initialized for experiment: {experiment_dir}",
        )

    def _setup_standardized_config_storage(self) -> None:
        """Initialize the StandardizedConfigStorage for configuration
        management.

        Implements action item 10 from subtask 9.3: Add validation to prevent
        training without proper configuration.
        """
        # Get experiment directory for configuration storage
        experiment_dir = (
            getattr(self.experiment_manager, "experiment_dir", None)
            if self.experiment_manager
            else None
        )

        # Fallback to checkpoint_dir parent if experiment_dir is not available
        if experiment_dir is None:
            experiment_dir = self.checkpoint_dir.parent

        # Initialize configuration storage
        config_storage_dir = experiment_dir / "configurations"
        self.config_storage = StandardizedConfigStorage(
            base_dir=config_storage_dir,
            include_environment=True,
            validate_on_save=True,
        )

        # Validate current configuration completeness
        validation_result = validate_configuration_completeness(
            self.full_cfg, strict=False
        )

        if not validation_result["is_valid"]:
            missing_required = validation_result["missing_required"]
            safe_log(
                self.internal_logger,
                "warning",
                f"Configuration validation found missing required fields: "
                f"{missing_required}",
            )

            # This implements the "prevent training without proper
            # configuration" requirement
            if validation_result.get("has_critical_missing", False):
                raise ValueError(
                    f"Training cannot proceed with incomplete configuration. "
                    f"Missing critical required fields: {missing_required}"
                )

        # Save the standardized configuration at initialization
        experiment_id = getattr(
            self.experiment_manager, "experiment_id", "default_experiment"
        )
        try:
            config_path = self.config_storage.save_configuration(
                config=self.full_cfg,
                experiment_id=experiment_id,
                config_name="training_config",
                format_type="yaml",
            )
            safe_log(
                self.internal_logger,
                "info",
                f"Training configuration saved to: {config_path}",
            )
        except Exception as e:
            safe_log(
                self.internal_logger,
                "warning",
                f"Failed to save initial training configuration: {e}",
            )

    def _setup_device_and_model(self):
        """Sets up the device and moves the model to it."""
        self.device = get_device(self.device_str)
        self.model.to(self.device)

    def _setup_optimizer_and_scheduler(self):
        """Sets up the optimizer and learning rate scheduler."""
        self.optimizer = create_optimizer(self.model, self.cfg.optimizer)
        self.scheduler = (
            create_lr_scheduler(self.optimizer, self.cfg.lr_scheduler)
            if "lr_scheduler" in self.cfg
            else None
        )

    def _setup_mixed_precision(self):
        """Sets up the gradient scaler for mixed-precision training."""
        self.scaler = (
            GradScaler(enabled=self.use_amp) if self.use_amp else None
        )

    def _load_checkpoint_state(self):
        """Loads the checkpoint state if a path is provided."""
        if self.checkpoint_load_path:
            checkpoint_data = load_checkpoint(
                checkpoint_path=self.checkpoint_load_path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
            )
            self.start_epoch = (
                int(checkpoint_data.get("epoch", self.start_epoch)) + 1
            )
            self.best_metric_value = float(
                checkpoint_data.get(
                    "best_metric_value", self.best_metric_value
                )
            )

    def _setup_early_stopping_instance(
        self, early_stopper_arg: EarlyStopping | None
    ):
        """Sets up the early stopping instance."""
        if early_stopper_arg is not None:
            self.early_stopper = early_stopper_arg
        else:
            self.early_stopper = setup_early_stopping(
                cfg=self.cfg,
                monitor_metric=self.monitor_metric,
                monitor_mode=self.monitor_mode,
                verbose=self.verbose,
                logger=self.internal_logger,
            )

        if self.early_stopper and self.early_stopper.enabled:
            # Override monitor_metric and monitor_mode from the early_stopper
            # config
            self.monitor_metric = self.early_stopper.monitor_metric
            self.monitor_mode = self.early_stopper.monitor_mode

    def _log_initialization_summary(self):
        """Logs a summary of the trainer's initialization."""
        safe_log(
            self.internal_logger,
            "info",
            "Trainer initialized.",
            {
                "Epochs": self.epochs,
                "Device": self.device,
                "AMP": self.use_amp,
                "Grad Accumulation": self.grad_accum_steps,
            },
        )

    def train(self) -> dict[str, float]:
        """Runs the full training loop starting from self.start_epoch."""
        start_msg = f"Starting training from epoch {self.start_epoch}..."
        safe_log(self.internal_logger, "info", start_msg)
        self.callback_handler.on_train_begin()
        start_time = time.time()
        final_val_results = {}

        for epoch in range(self.start_epoch, int(self.epochs) + 1):
            self.callback_handler.on_epoch_begin(epoch)
            epoch_start_time = time.time()
            safe_log(
                self.internal_logger,
                "info",
                f"--- Epoch {epoch}/{self.epochs} ---",
            )

            train_loss = self._train_epoch(epoch)
            val_results = self.validate(epoch)
            final_val_results = val_results  # Keep track of last results

            # --- Handle LR Scheduling ---
            current_lr = None
            if self.scheduler:
                current_lr = self._step_scheduler(val_results)
                if current_lr is not None and self.logger_instance:
                    self.logger_instance.log_scalar(
                        tag="lr", value=current_lr, step=epoch
                    )

            # Log epoch summary using MetricsManager
            train_metrics = {"loss": train_loss}
            self.metrics_manager.log_epoch_summary(
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics={
                    k.replace("val_", ""): v for k, v in val_results.items()
                },
                learning_rate=current_lr,
            )

            # --- Checkpointing Logic (refactorizado) ---
            checkpoint_context = CheckpointContext(
                epoch=epoch,
                val_results=val_results,
                monitor_metric=self.monitor_metric,
                monitor_mode=self.monitor_mode,
                best_metric_value=self.best_metric_value,
            )
            checkpoint_config = CheckpointConfig(
                checkpoint_dir=str(self.checkpoint_dir),
                logger=self.internal_logger,
                keep_last_n=self.cfg.get(
                    "checkpoints.keep_last_n", 1
                ),  # Default from config or 1
                last_filename=self.cfg.get(
                    "checkpoints.last_filename", "checkpoint_last.pth"
                ),
                best_filename=self.best_filename,
            )

            self.best_metric_value = handle_epoch_checkpointing(
                context=checkpoint_context,
                config=checkpoint_config,
                model=self.model,
                optimizer=self.optimizer,
            )

            # Save configuration alongside checkpoints (action item 4 from
            # subtask 9.3)
            if hasattr(self, "config_storage"):
                try:
                    experiment_id = getattr(
                        self.experiment_manager,
                        "experiment_id",
                        "default_experiment",
                    )

                    # Save configuration with epoch information
                    config_filename = f"config_epoch_{epoch:04d}"
                    self.config_storage.save_configuration(
                        config=self.full_cfg,
                        experiment_id=experiment_id,
                        config_name=config_filename,
                        format_type="yaml",
                    )

                    # If this is the best model, also save as "best_config"
                    if self._check_if_best(val_results):
                        self.config_storage.save_configuration(
                            config=self.full_cfg,
                            experiment_id=experiment_id,
                            config_name="best_config",
                            format_type="yaml",
                        )
                        safe_log(
                            self.internal_logger,
                            "info",
                            "Best model configuration saved alongside "
                            "checkpoint",
                        )

                except Exception as e:
                    safe_log(
                        self.internal_logger,
                        "warning",
                        f"Failed to save configuration for epoch {epoch}: {e}",
                    )

            # --- Early Stopping Check ---
            if self.early_stopper is not None and getattr(
                self.early_stopper, "enabled", False
            ):
                current_metric = val_results.get(self.monitor_metric)
                if self.early_stopper.should_stop(current_metric):
                    safe_log(
                        self.internal_logger,
                        "info",
                        "Early stopping triggered.",
                    )
                    break  # Exit training loop

            self.callback_handler.on_epoch_end(epoch, logs=val_results)
            epoch_duration = time.time() - epoch_start_time
            safe_log(
                self.internal_logger,
                "info",
                f"Epoch {epoch} finished in {epoch_duration:.2f}s",
            )

        total_time = time.time() - start_time
        safe_log(
            self.internal_logger,
            "info",
            f"Training finished in {total_time:.2f}s",
        )

        # Export final metrics summary
        summary_path = self.metrics_manager.export_metrics_summary()
        self.callback_handler.on_train_end()
        safe_log(
            self.internal_logger,
            "info",
            f"Final metrics summary exported to: {summary_path}",
        )

        return final_val_results

    def _train_epoch(self, epoch: int) -> float:
        """Runs a single training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        log_interval = self.cfg.get("log_interval_batches", 0)
        self.optimizer.zero_grad()
        for batch_idx, batch in enumerate(self.train_loader):
            self.callback_handler.on_batch_begin(batch_idx)
            with amp_autocast(self.use_amp):
                metrics = train_step(
                    model=self.model,
                    batch=batch,
                    loss_fn=self.loss_fn,
                    optimizer=self.optimizer,
                    device=self.device,
                    metrics_dict=self.metrics_dict,
                )
                batch_loss = metrics["loss"]
            # Convert loss to tensor if needed
            if not isinstance(batch_loss, torch.Tensor):
                batch_loss = torch.tensor(batch_loss)
            total_loss += batch_loss.item()
            optimizer_step_with_accumulation(
                optimizer=self.optimizer,
                scaler=self.scaler,
                loss=batch_loss,
                grad_accum_steps=self.grad_accum_steps,
                batch_idx=batch_idx,
                use_amp=self.use_amp,
            )
            self.callback_handler.on_batch_end(
                batch_idx, logs={"loss": batch_loss.item()}
            )
            if (
                self.logger_instance
                and log_interval > 0
                and (batch_idx + 1) % log_interval == 0
            ):
                global_step = (epoch - 1) * num_batches + batch_idx + 1
                self.logger_instance.log_scalar(
                    tag="train_batch/batch_loss",
                    value=batch_loss.item(),
                    step=global_step,
                )
        avg_loss = total_loss / num_batches
        log_msg = f"Epoch {epoch} | Average Training Loss: {avg_loss:.4f}"
        safe_log(self.internal_logger, "info", log_msg)
        if self.logger_instance:
            self.logger_instance.log_scalar(
                tag="train/epoch_loss", value=avg_loss, step=epoch
            )
        return avg_loss

    def validate(self, epoch: int) -> dict[str, float]:
        """Runs validation over the val_loader and aggregates metrics."""
        self.model.eval()
        total_metrics = {"loss": 0.0}
        total_metrics.update(dict.fromkeys(self.metrics_dict, 0.0))
        num_batches = len(self.val_loader)
        with torch.no_grad():
            for batch in self.val_loader:
                metrics = val_step(
                    model=self.model,
                    batch=batch,
                    loss_fn=self.loss_fn,
                    device=self.device,
                    metrics_dict=self.metrics_dict,
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
        self.metrics_manager.log_training_metrics(
            epoch=epoch,
            step=0,  # Step 0 for epoch-level validation
            metrics=val_metrics,
            phase="val",
        )

        # Legacy logging for backward compatibility
        log_validation_results(self.internal_logger, epoch, val_metrics)

        # Original logger instance logging (if available)
        if self.logger_instance:
            for name, value in val_metrics.items():
                self.logger_instance.log_scalar(
                    tag=f"{name}", value=value, step=epoch
                )
        return val_metrics

    # --- Helper Methods ---
    def _step_scheduler(self, metrics=None) -> float | None:
        """Steps the learning rate scheduler using a helper utility."""
        return step_scheduler_helper(
            scheduler=self.scheduler,
            optimizer=self.optimizer,
            monitor_metric=self.monitor_metric,
            metrics=metrics,
            logger=self.internal_logger,
        )

    def _check_if_best(self, metrics: dict[str, float]) -> bool:
        """Checks if the current model is the best one based on the monitored
        metric."""
        if not self.save_best_enabled:
            return False

        current_metric_value = metrics.get(self.monitor_metric)
        if current_metric_value is None:
            return False

        if self.monitor_mode == "min":
            return current_metric_value < self.best_metric_value
        return current_metric_value > self.best_metric_value


# --- Remove old standalone functions ---
# def train_one_epoch(...): ...
# def evaluate(...): ...
