"""Trainer setup component.

Handles the setup of various trainer components including monitoring, checkpointing,
metrics, and device configuration.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

from crackseg.training.factory import create_lr_scheduler, create_optimizer
from crackseg.utils.core.device import get_device
from crackseg.utils.experiment.manager import ExperimentManager
from crackseg.utils.logging.base import get_logger
from crackseg.utils.logging.metrics_manager import MetricsManager
from crackseg.utils.logging.training import safe_log
from crackseg.utils.monitoring import (
    BaseCallback,
    CallbackHandler,
    MonitoringManager,
)
from crackseg.utils.storage import (
    StandardizedConfigStorage,
    setup_checkpointing,
    validate_configuration_completeness,
)
from crackseg.utils.training.amp_utils import GradScaler
from crackseg.utils.training.early_stopping import EarlyStopping
from crackseg.utils.training.early_stopping_setup import setup_early_stopping


class TrainerSetup:
    """Handles trainer component setup and configuration."""

    def __init__(self) -> None:
        """Initialize the trainer setup component."""
        pass

    def setup_monitoring(
        self, trainer_instance: Any, callbacks: list[BaseCallback] | None
    ) -> None:
        """Initializes the monitoring and callback system."""
        trainer_instance.monitoring_manager = MonitoringManager()
        trainer_instance.callback_handler = CallbackHandler(
            callbacks or [], trainer_instance.monitoring_manager
        )

    def setup_checkpointing_attributes(self, trainer_instance: Any) -> None:
        """Sets up attributes related to checkpointing."""
        (checkpoint_dir_str, trainer_instance.experiment_manager) = (
            setup_checkpointing(
                trainer_instance.full_cfg,
                getattr(
                    trainer_instance.logger_instance,
                    "experiment_manager",
                    None,
                ),
                trainer_instance.internal_logger,
            )
        )

        # Convert string to Path object
        trainer_instance.checkpoint_dir = Path(checkpoint_dir_str)

        # Auto-detect experiment information from configuration
        self._auto_detect_experiment_info(trainer_instance)

        trainer_instance.save_freq = trainer_instance.cfg.get("save_freq", 0)
        trainer_instance.checkpoint_load_path = trainer_instance.cfg.get(
            "checkpoint_load_path", None
        )

        save_best_config = trainer_instance.cfg.get("save_best", {})
        # Fallback to checkpoints.save_best if training.save_best is not present
        if not save_best_config and "checkpoints" in trainer_instance.cfg:
            checkpoints_cfg = trainer_instance.cfg.get("checkpoints", {})
            save_best_config = checkpoints_cfg.get("save_best", {})

        # Handle case where save_best is a boolean instead of dict
        if isinstance(save_best_config, bool):
            save_best_config = {"enabled": save_best_config}

        trainer_instance.save_best_enabled = save_best_config.get(
            "enabled", False
        )
        # First try to get from experiment config, then from save_best_config, then defaults
        trainer_instance.monitor_metric = str(
            trainer_instance.cfg.get(
                "monitor_metric",
                save_best_config.get("monitor_metric", "val_loss"),
            )
        )
        trainer_instance.monitor_mode = str(
            trainer_instance.cfg.get(
                "monitor_mode", save_best_config.get("monitor_mode", "min")
            )
        )
        trainer_instance.best_filename = str(
            save_best_config.get("best_filename", "model_best.pth.tar")
        )
        trainer_instance.best_metric_value = (
            float("inf")
            if trainer_instance.monitor_mode == "min"
            else float("-inf")
        )

        # Initialize MetricsManager for unified metric logging
        self._setup_metrics_manager(trainer_instance)

        # Initialize StandardizedConfigStorage for configuration management
        self._setup_standardized_config_storage(trainer_instance)

    def _auto_detect_experiment_info(self, trainer_instance: Any) -> None:
        """Auto-detect experiment information from configuration and create proper experiment_manager."""
        # Extract experiment information from configuration
        experiment_name = "default_experiment"
        base_dir = "artifacts"
        timestamp = None
        experiment_dir = None

        # Log configuration structure for troubleshooting
        if hasattr(trainer_instance.full_cfg, "keys"):
            available_keys = list(trainer_instance.full_cfg.keys())
            safe_log(
                trainer_instance.internal_logger,
                "debug",
                f"Available config keys: {available_keys}",
            )

        # Try to get experiment configuration from different locations
        experiment_config = None

        # First try: nested experiments.swinv2_hybrid.experiment (most specific)
        if hasattr(trainer_instance.full_cfg, "experiments"):
            experiments = trainer_instance.full_cfg.experiments
            if hasattr(experiments, "swinv2_hybrid"):
                swinv2_config = experiments.swinv2_hybrid
                if hasattr(swinv2_config, "experiment"):
                    experiment_config = swinv2_config.experiment
                    safe_log(
                        trainer_instance.internal_logger,
                        "info",
                        "Found experiment config in experiments.swinv2_hybrid.experiment",
                    )

        # Second try: direct experiment key (fallback)
        if experiment_config is None and (
            hasattr(trainer_instance.full_cfg, "experiment")
            and trainer_instance.full_cfg.experiment
        ):
            experiment_config = trainer_instance.full_cfg.experiment
            safe_log(
                trainer_instance.internal_logger,
                "info",
                "Found experiment config at root level (fallback)",
            )

        if experiment_config:
            # Get experiment name
            if hasattr(experiment_config, "name"):
                experiment_name = experiment_config.name

            # Use the pre-resolved output_dir from Hydra (includes timestamp)
            if hasattr(experiment_config, "output_dir"):
                output_dir = experiment_config.output_dir
                if isinstance(output_dir, str):
                    experiment_dir = Path(output_dir)
                    # Extract timestamp from the directory name
                    dir_name = experiment_dir.name
                    if "-" in dir_name:
                        # Format: "20240804-154327-experiment_name"
                        parts = dir_name.split("-", 2)
                        if len(parts) >= 2:
                            timestamp = f"{parts[0]}-{parts[1]}"

                    # Set base_dir from the parent directories
                    if "artifacts/experiments" in str(experiment_dir):
                        base_dir = "artifacts"

        # Fallback: create timestamp if not found in config
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        # Fallback: create experiment_dir if not found in config
        if experiment_dir is None:
            experiment_dir = Path(
                f"artifacts/experiments/{timestamp}-{experiment_name}"
            )

        experiment_id = f"{timestamp}-{experiment_name}"

        # Create the experiment directory if it doesn't exist
        experiment_dir.mkdir(parents=True, exist_ok=True)

        # Create ExperimentManager with the correct information
        trainer_instance.experiment_manager = ExperimentManager(
            base_dir=base_dir,
            experiment_name=experiment_name,
            config=trainer_instance.full_cfg,
            create_dirs=True,
            timestamp=timestamp,
        )

        # Ensure the experiment_manager uses the correct experiment_dir
        trainer_instance.experiment_manager.experiment_dir = experiment_dir
        trainer_instance.experiment_manager.experiment_id = experiment_id

        # Update checkpoint directory to use the experiment-specific directory
        trainer_instance.checkpoint_dir = (
            trainer_instance.experiment_manager.get_path("checkpoints")
        )

        safe_log(
            trainer_instance.internal_logger,
            "info",
            f"Auto-detected experiment: {experiment_id}",
        )
        safe_log(
            trainer_instance.internal_logger,
            "info",
            f"Experiment directory: {trainer_instance.experiment_manager.experiment_dir}",
        )

    def _setup_metrics_manager(self, trainer_instance: Any) -> None:
        """Initialize the MetricsManager for standardized metric logging."""
        # Get experiment directory from experiment_manager or checkpoint_dir
        experiment_dir = (
            getattr(
                trainer_instance.experiment_manager, "experiment_dir", None
            )
            if trainer_instance.experiment_manager
            else None
        )

        # Fallback to checkpoint_dir parent if experiment_dir is not available
        if experiment_dir is None:
            experiment_dir = trainer_instance.checkpoint_dir.parent

        # Use get_logger for Python logger instead of BaseLogger
        python_logger = get_logger("trainer.metrics")

        trainer_instance.metrics_manager = MetricsManager(
            experiment_dir=experiment_dir,
            logger=python_logger,
            config=trainer_instance.full_cfg,
        )

        safe_log(
            trainer_instance.internal_logger,
            "info",
            f"MetricsManager initialized for experiment: {experiment_dir}",
        )

    def _setup_standardized_config_storage(
        self, trainer_instance: Any
    ) -> None:
        """Initialize the StandardizedConfigStorage for configuration management."""
        # Get experiment directory for configuration storage
        experiment_dir = (
            getattr(
                trainer_instance.experiment_manager, "experiment_dir", None
            )
            if trainer_instance.experiment_manager
            else None
        )

        # Fallback to checkpoint_dir parent if experiment_dir is not available
        if experiment_dir is None:
            experiment_dir = trainer_instance.checkpoint_dir.parent

        # Initialize configuration storage
        config_storage_dir = experiment_dir / "configurations"
        trainer_instance.config_storage = StandardizedConfigStorage(
            base_dir=config_storage_dir,
            include_environment=True,
            validate_on_save=True,
        )

        # Validate current configuration completeness
        validation_result = validate_configuration_completeness(
            trainer_instance.full_cfg, strict=False
        )

        if not validation_result["is_valid"]:
            missing_required = validation_result["missing_required"]
            safe_log(
                trainer_instance.internal_logger,
                "warning",
                f"Configuration validation found missing required fields: "
                f"{missing_required}",
            )

            # This implements the "prevent training without proper configuration" requirement
            if validation_result.get("has_critical_missing", False):
                raise ValueError(
                    f"Training cannot proceed with incomplete configuration. "
                    f"Missing critical required fields: {missing_required}"
                )

        # Save the standardized configuration at initialization
        experiment_id = getattr(
            trainer_instance.experiment_manager,
            "experiment_id",
            "default_experiment",
        )
        try:
            config_path = trainer_instance.config_storage.save_configuration(
                config=trainer_instance.full_cfg,
                experiment_id=experiment_id,
                config_name="training_config",
                format_type="yaml",
            )
            safe_log(
                trainer_instance.internal_logger,
                "info",
                f"Training configuration saved to: {config_path}",
            )
        except Exception as e:
            safe_log(
                trainer_instance.internal_logger,
                "warning",
                f"Failed to save initial training configuration: {e}",
            )

    def setup_device_and_model(self, trainer_instance: Any) -> None:
        """Sets up the device and moves the model to it."""
        trainer_instance.device = get_device(trainer_instance.device_str)
        trainer_instance.model.to(trainer_instance.device)

    def setup_optimizer_and_scheduler(self, trainer_instance: Any) -> None:
        """Sets up the optimizer and learning rate scheduler."""
        trainer_instance.optimizer = create_optimizer(
            trainer_instance.model.parameters(), trainer_instance.cfg.optimizer
        )
        trainer_instance.scheduler = (
            create_lr_scheduler(
                trainer_instance.optimizer, trainer_instance.cfg.lr_scheduler
            )
            if "lr_scheduler" in trainer_instance.cfg
            else None
        )

    def setup_mixed_precision(self, trainer_instance: Any) -> None:
        """Sets up the gradient scaler for mixed-precision training."""
        trainer_instance.scaler = (
            GradScaler(enabled=trainer_instance.use_amp)
            if trainer_instance.use_amp
            else None
        )

    def load_checkpoint_state(self, trainer_instance: Any) -> None:
        """Loads the checkpoint state if a path is provided."""
        if trainer_instance.checkpoint_load_path:
            from crackseg.utils.storage import load_checkpoint

            checkpoint_data = load_checkpoint(
                checkpoint_path=trainer_instance.checkpoint_load_path,
                model=trainer_instance.model,
                optimizer=trainer_instance.optimizer,
                scheduler=trainer_instance.scheduler,
            )
            trainer_instance.start_epoch = (
                int(checkpoint_data.get("epoch", trainer_instance.start_epoch))
                + 1
            )
            trainer_instance.best_metric_value = float(
                checkpoint_data.get(
                    "best_metric_value", trainer_instance.best_metric_value
                )
            )

    def setup_early_stopping_instance(
        self, trainer_instance: Any, early_stopper: EarlyStopping | None
    ) -> None:
        """Sets up the early stopping instance."""
        if early_stopper is not None:
            trainer_instance.early_stopper = early_stopper
        else:
            trainer_instance.early_stopper = setup_early_stopping(
                cfg=trainer_instance.cfg,
                monitor_metric=trainer_instance.monitor_metric,
                monitor_mode=trainer_instance.monitor_mode,
                verbose=trainer_instance.verbose,
                logger=trainer_instance.internal_logger,
            )

        if (
            trainer_instance.early_stopper
            and trainer_instance.early_stopper.enabled
        ):
            # Override monitor_metric and monitor_mode from the early_stopper config
            trainer_instance.monitor_metric = (
                trainer_instance.early_stopper.monitor_metric
            )
            trainer_instance.monitor_mode = (
                trainer_instance.early_stopper.monitor_mode
            )

    def log_initialization_summary(self, trainer_instance: Any) -> None:
        """Logs a summary of the trainer's initialization."""
        safe_log(
            trainer_instance.internal_logger,
            "info",
            "Trainer initialized.",
            {
                "Epochs": trainer_instance.epochs,
                "Device": trainer_instance.device,
                "AMP": trainer_instance.use_amp,
                "Grad Accumulation": trainer_instance.grad_accum_steps,
            },
        )
