"""Helper functions extracted from trainer setup to reduce module size."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from crackseg.utils.logging.base import get_logger
from crackseg.utils.logging.metrics_manager import MetricsManager
from crackseg.utils.logging.training import safe_log
from crackseg.utils.storage import (
    StandardizedConfigStorage,
    validate_configuration_completeness,
)


def setup_metrics_manager(trainer_instance: Any) -> None:
    experiment_dir = (
        getattr(trainer_instance.experiment_manager, "experiment_dir", None)
        if trainer_instance.experiment_manager
        else None
    )
    if experiment_dir is None:
        experiment_dir = trainer_instance.checkpoint_dir.parent

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


def setup_standardized_config_storage(trainer_instance: Any) -> None:
    experiment_dir = (
        getattr(trainer_instance.experiment_manager, "experiment_dir", None)
        if trainer_instance.experiment_manager
        else None
    )
    if experiment_dir is None:
        experiment_dir = trainer_instance.checkpoint_dir.parent

    config_storage_dir = Path(experiment_dir) / "configurations"
    trainer_instance.config_storage = StandardizedConfigStorage(
        base_dir=config_storage_dir,
        include_environment=True,
        validate_on_save=True,
    )

    validation_result = validate_configuration_completeness(
        trainer_instance.full_cfg, strict=False
    )
    if not validation_result["is_valid"]:
        missing_required = validation_result["missing_required"]
        safe_log(
            trainer_instance.internal_logger,
            "warning",
            "Configuration validation found missing required fields: "
            f"{missing_required}",
        )
        if validation_result.get("has_critical_missing", False):
            raise ValueError(
                "Training cannot proceed with incomplete configuration. "
                f"Missing critical required fields: {missing_required}"
            )

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
    except Exception as exc:  # noqa: BLE001
        safe_log(
            trainer_instance.internal_logger,
            "warning",
            f"Failed to save initial training configuration: {exc}",
        )
