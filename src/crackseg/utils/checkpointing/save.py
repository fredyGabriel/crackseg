"""Checkpoint saving functionality.

This module handles the saving of model checkpoints with standardized
format and validation.
"""

import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from .config import (
    CheckpointSaveConfig,
    CheckpointSpec,
    generate_checkpoint_metadata,
)

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    config: CheckpointSaveConfig,
    scheduler: LRScheduler | None = None,
    best_metric_value: float | None = None,
    metrics: dict[str, float] | None = None,
    training_config: dict[str, Any] | None = None,
) -> None:
    """Save a model checkpoint with standardized format and validation.

    Args:
        model: PyTorch model to save
        optimizer: Optimizer to save state from
        epoch: Current training epoch
        config: Checkpoint save configuration
        scheduler: Optional learning rate scheduler
        best_metric_value: Optional best metric value for tracking
        metrics: Optional current epoch metrics
        training_config: Optional training configuration
    """
    checkpoint_dir = _ensure_checkpoint_dir(
        Path(config.checkpoint_dir), logger
    )
    checkpoint_path = checkpoint_dir / config.filename

    # Build standardized checkpoint data
    checkpoint_data: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    # Add automatic metadata
    if config.include_python_info:
        checkpoint_data.update(generate_checkpoint_metadata())
    else:
        # Minimal required metadata
        checkpoint_data.update(
            {
                "pytorch_version": torch.__version__,
                "timestamp": datetime.now().isoformat(),
            }
        )

    # Add scheduler state if available and requested
    if scheduler is not None and config.include_scheduler:
        checkpoint_data["scheduler_state_dict"] = scheduler.state_dict()

    # Add optional data
    if best_metric_value is not None:
        checkpoint_data["best_metric_value"] = best_metric_value

    if metrics is not None:
        checkpoint_data["metrics"] = metrics

    if training_config is not None:
        checkpoint_data["config"] = training_config

    # Add any additional data from config
    if config.additional_data:
        checkpoint_data.update(config.additional_data)

    # Validate completeness if requested
    if config.validate_completeness:
        spec = CheckpointSpec()
        is_valid, missing_fields = validate_checkpoint_completeness(
            checkpoint_data, spec
        )
        if not is_valid:
            # Log warning but don't fail - some fields might be intentionally
            # omitted
            logger.warning(
                f"Checkpoint missing recommended fields: {missing_fields}. "
                "Continuing with save."
            )

    _save_to_path(checkpoint_data, checkpoint_path, epoch, logger)
    _cleanup_old_checkpoints(checkpoint_dir, config.keep_last_n, logger)


def create_standardized_filename(
    base_name: str,
    epoch: int,
    timestamp: str | None = None,
    is_best: bool = False,
) -> str:
    """Create standardized checkpoint filename with consistent naming pattern.

    Args:
        base_name: Base name for the checkpoint (e.g., 'checkpoint', 'model')
        epoch: Training epoch number
        timestamp: Optional timestamp string, auto-generated if None
        is_best: Whether this is the best model checkpoint

    Returns:
        Standardized filename string
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if is_best:
        return f"{base_name}_best_epoch_{epoch:03d}_{timestamp}.pth"
    else:
        return f"{base_name}_epoch_{epoch:03d}_{timestamp}.pth"


def _ensure_checkpoint_dir(
    requested_dir: Path, logger_instance: logging.Logger
) -> Path:
    """Ensures checkpoint directory exists, with fallback logic.

    Args:
        requested_dir: Requested checkpoint directory
        logger_instance: Logger instance for logging

    Returns:
        Path to the checkpoint directory
    """
    try:
        requested_dir.mkdir(parents=True, exist_ok=True)
        logger_instance.info(f"Checkpoint directory verified: {requested_dir}")
        return requested_dir
    except (PermissionError, OSError) as e:
        logger_instance.error(
            f"Failed to create checkpoint directory {requested_dir}: {e}"
        )
        fallback_dir = Path("./checkpoints")
        try:
            fallback_dir.mkdir(parents=True, exist_ok=True)
            logger_instance.warning(
                f"Using fallback checkpoint directory: {fallback_dir}"
            )
            return fallback_dir
        except (PermissionError, OSError) as e_fallback:
            logger_instance.error(
                f"Failed to create fallback directory {fallback_dir}: "
                f"{e_fallback}"
            )
            raise RuntimeError(
                f"Cannot create checkpoint directory: {e_fallback}"
            ) from e_fallback


def _save_to_path(
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    epoch: int,
    logger_instance: logging.Logger,
) -> None:
    """Verifies writability and saves checkpoint to the given path.

    Args:
        checkpoint: Checkpoint data to save
        checkpoint_path: Path where to save the checkpoint
        epoch: Current epoch number
        logger_instance: Logger instance for logging
    """
    try:
        test_file = checkpoint_path.parent / f"test_write_{epoch}.tmp"
        test_file.touch(exist_ok=True)
        test_file.unlink()
    except (PermissionError, OSError) as e:
        logger_instance.error(
            f"Checkpoint directory {checkpoint_path.parent} is not writable: "
            f"{e}"
        )
        raise RuntimeError(
            f"Checkpoint directory {checkpoint_path.parent} is not writable: "
            f"{e}"
        ) from e

    try:
        torch.save(checkpoint, checkpoint_path)
        logger_instance.info(
            f"Saved checkpoint at epoch {epoch} to {checkpoint_path}"
        )
    except (OSError, RuntimeError, pickle.PicklingError) as e:
        logger_instance.error(
            f"Failed to save checkpoint to {checkpoint_path}: {e}"
        )
        # Try an alternative path as fallback
        alt_path = (
            checkpoint_path.parent / f"emergency_checkpoint_epoch_{epoch}.pt"
        )
        try:
            torch.save(checkpoint, alt_path)
            logger_instance.warning(
                f"Saved emergency checkpoint to {alt_path}"
            )
        except (OSError, RuntimeError, pickle.PicklingError) as e2:
            logger_instance.error(f"Failed to save emergency checkpoint: {e2}")
            raise RuntimeError(f"Cannot save checkpoint: {e2}") from e2


def _cleanup_old_checkpoints(
    checkpoint_dir: Path, keep_last_n: int, logger_instance: logging.Logger
) -> None:
    """Cleans up old checkpoints based on filename epoch number.

    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last_n: Number of recent checkpoints to keep
        logger_instance: Logger instance for logging
    """
    if keep_last_n <= 0:
        return

    prefix = "_epoch_"
    suffix = ".pth"
    checkpoint_files: list[tuple[int, Path]] = []
    glob_pattern = f"*{prefix}*{suffix}"

    for f_path in checkpoint_dir.glob(glob_pattern):
        try:
            epoch_str = f_path.stem.split(prefix)[-1]
            file_epoch = int(epoch_str)
            checkpoint_files.append((file_epoch, f_path))
        except (IndexError, ValueError):
            log_msg = (
                f"Could not parse epoch from {f_path.name}, "
                f"skipping cleanup for this file."
            )
            logger_instance.warning(log_msg)
            continue

    checkpoint_files.sort(key=lambda x: x[0])

    if len(checkpoint_files) > keep_last_n:
        files_to_remove = checkpoint_files[:-keep_last_n]
        logger_instance.debug(
            f"Removing {len(files_to_remove)} old checkpoints"
        )
        for file_epoch, f_path_to_remove in files_to_remove:
            try:
                f_path_to_remove.unlink()
                rm_msg = (
                    f"Removed old ckpt (epoch {file_epoch}): "
                    f"{f_path_to_remove.name}"
                )
                logger_instance.debug(rm_msg)
            except OSError as e:
                err_msg = (
                    f"Error removing old checkpoint {f_path_to_remove}: {e}"
                )
                logger_instance.error(err_msg)


def validate_checkpoint_completeness(
    checkpoint_data: dict[str, Any], spec: CheckpointSpec | None = None
) -> tuple[bool, list[str]]:
    """Validate that checkpoint contains all required fields.

    Args:
        checkpoint_data: Checkpoint dictionary to validate
        spec: Specification defining required fields

    Returns:
        Tuple of (is_valid, missing_fields)
    """
    if spec is None:
        spec = CheckpointSpec()

    missing_fields = []
    for field in spec.required_fields:
        if field not in checkpoint_data:
            missing_fields.append(field)

    return len(missing_fields) == 0, missing_fields
