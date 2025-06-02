"""Checkpoint management utilities."""

import pickle  # Import pickle for specific exceptions
import platform
import sys
import zipfile  # Import zipfile for specific exceptions
from dataclasses import dataclass, field  # Import dataclass
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CheckpointSpec:
    """Specification defining required and optional contents of a checkpoint.

    This serves as documentation and validation schema for checkpoint format.
    """

    # Required fields for complete model restoration
    required_fields: set[str] = field(
        default_factory=lambda: {
            "epoch",
            "model_state_dict",
            "optimizer_state_dict",
            "pytorch_version",
            "timestamp",
            "config",
        }
    )

    # Optional fields that enhance checkpoint utility
    optional_fields: set[str] = field(
        default_factory=lambda: {
            "scheduler_state_dict",
            "best_metric_value",
            "metrics",
            "python_version",
            "experiment_id",
            "git_commit",
            "notes",
        }
    )

    # Fields that should be automatically generated
    auto_generated_fields: set[str] = field(
        default_factory=lambda: {
            "pytorch_version",
            "python_version",
            "timestamp",
        }
    )


@dataclass
class CheckpointSaveConfig:
    """Configuration for saving a checkpoint with enhanced validation."""

    checkpoint_dir: str | Path
    filename: str = "checkpoint.pt"
    additional_data: dict[str, Any] | None = None
    keep_last_n: int = 1
    include_scheduler: bool = True
    include_python_info: bool = True
    validate_completeness: bool = True


def generate_checkpoint_metadata() -> dict[str, Any]:
    """Generate automatic metadata for checkpoint."""
    return {
        "pytorch_version": torch.__version__,
        "python_version": (
            f"{sys.version_info.major}.{sys.version_info.minor}."
            f"{sys.version_info.micro}"
        ),
        "platform": platform.platform(),
        "timestamp": datetime.now().isoformat(),
    }


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
    for field_name in spec.required_fields:
        if field_name not in checkpoint_data:
            missing_fields.append(field_name)

    is_valid = len(missing_fields) == 0
    return is_valid, missing_fields


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


# Helper function to manage directory creation with fallback
def _ensure_checkpoint_dir(
    requested_dir: Path, logger_instance: Logger
) -> Path:
    """Ensures checkpoint directory exists, with fallback logic."""
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


# Helper function to verify writability and save checkpoint
def _save_to_path(
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    epoch: int,
    logger_instance: Logger,
) -> None:
    """Verifies writability and saves checkpoint to the given path."""
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


# Helper function to clean up old checkpoints
def _cleanup_old_checkpoints(
    checkpoint_dir: Path, keep_last_n: int, logger_instance: Logger
) -> None:
    """Cleans up old checkpoints based on filename epoch number."""
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


def verify_checkpoint_integrity(
    checkpoint_path: str | Path, spec: CheckpointSpec | None = None
) -> dict[str, Any]:
    """Verify checkpoint integrity and completeness.

    Args:
        checkpoint_path: Path to checkpoint file
        spec: Specification for validation

    Returns:
        Dictionary with verification results
    """
    if spec is None:
        spec = CheckpointSpec()

    checkpoint_path = Path(checkpoint_path)

    # Initialize validation variables
    is_valid = False
    missing_fields: list[str] = []

    verification_result: dict[str, Any] = {
        "path": str(checkpoint_path),
        "exists": checkpoint_path.exists(),
        "is_valid": False,
        "missing_fields": [],
        "size_bytes": 0,
        "error": None,
    }

    if not checkpoint_path.exists():
        verification_result["error"] = "File does not exist"
        return verification_result

    try:
        # Get file size
        verification_result["size_bytes"] = checkpoint_path.stat().st_size

        # Load and validate checkpoint
        checkpoint_data = torch.load(
            checkpoint_path, map_location=torch.device("cpu")
        )

        # Validate completeness
        is_valid, missing_fields = validate_checkpoint_completeness(
            checkpoint_data, spec
        )

        verification_result["is_valid"] = is_valid
        verification_result["missing_fields"] = missing_fields

        # Add metadata from checkpoint
        verification_result["epoch"] = checkpoint_data.get("epoch")
        verification_result["pytorch_version"] = checkpoint_data.get(
            "pytorch_version"
        )
        verification_result["timestamp"] = checkpoint_data.get("timestamp")

        logger.info(f"Checkpoint verification completed for {checkpoint_path}")

    except Exception as e:
        verification_result["error"] = f"Failed to load/verify checkpoint: {e}"
        logger.error(
            f"Checkpoint verification failed for {checkpoint_path}: {e}"
        )

    if not is_valid:
        logger.error(
            f"Checkpoint validation failed. Missing fields: {missing_fields}"
        )
        raise ValueError(
            f"Invalid checkpoint format. Missing required fields: "
            f"{missing_fields}"
        )

    return verification_result


@dataclass
class CheckpointLoadConfig:  # Renamed for clarity if needed, or keep as is
    """Configuration for loading a checkpoint (Not used in save_checkpoint)."""

    # Parameters relevant to loading, if any specific ones arise.
    # For now, load_checkpoint and load_checkpoint_dict manage their own args.
    pass


def load_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    device: torch.device | None = None,
    strict_validation: bool = False,
) -> dict[str, Any]:
    """Load a model checkpoint with enhanced validation and compatibility.

    Args:
        checkpoint_path: Path to the checkpoint file
        model: The PyTorch model to load weights into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Optional device to load the model to
        strict_validation: Whether to enforce strict checkpoint validation

    Returns:
        Dict containing checkpoint data (epoch and any additional data)

    Raises:
        FileNotFoundError: If checkpoint_path does not exist.
        KeyError: If 'model_state_dict' is missing in the checkpoint.
        Exception: Other errors during torch.load or state dict loading.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Determine map_location for torch.load
    map_location = device
    if map_location is None:
        try:
            map_location = model.device
        except AttributeError:
            map_location = torch.device("cpu")
            logger.debug(
                "Model has no .device attribute, loading checkpoint to CPU"
            )

    # Load checkpoint
    try:
        # Set weights_only=True for security if only loading state_dicts
        checkpoint = torch.load(checkpoint_path, map_location=device)
        log_msg = (
            f"Loading checkpoint from {checkpoint_path} to {map_location}"
        )
        logger.info(log_msg)
    except Exception as e:
        # torch.load can raise several types of exceptions depending on the
        # error, from I/O problems to deserialization errors
        logger.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")
        raise

    # Validate checkpoint if strict validation is enabled
    if strict_validation:
        spec = CheckpointSpec()
        is_valid, missing_fields = validate_checkpoint_completeness(
            checkpoint, spec
        )
        if not is_valid:
            logger.error(
                f"Checkpoint validation failed. Missing fields: "
                f"{missing_fields}"
            )
            raise ValueError(
                f"Invalid checkpoint format. Missing: {missing_fields}"
            )

    # Log checkpoint metadata
    if "pytorch_version" in checkpoint:
        logger.info(
            f"Checkpoint PyTorch version: {checkpoint['pytorch_version']}"
        )
    if "timestamp" in checkpoint:
        logger.info(f"Checkpoint timestamp: {checkpoint['timestamp']}")

    # Load model state
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"'model_state_dict' not found in {checkpoint_path}")
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
        # Move model to target device *after* loading state dict
        if device:
            model.to(device)
    except (RuntimeError, KeyError, ValueError, TypeError) as e:
        logger.error(f"Failed to load model_state_dict: {e}")
        raise

    # Load optimizer state if provided and available
    if optimizer:
        if "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                logger.debug("Loaded optimizer state.")
            except (RuntimeError, KeyError, ValueError, AttributeError) as e:
                err_msg = f"Failed to load optimizer_state_dict: {e}"
                logger.error(err_msg)
                # Decide if this should be a fatal error or just a warning
                # raise # Consider re-raising or handling differently
        else:
            warn_msg = (
                "Optimizer provided, but 'optimizer_state_dict' "
                "not found in checkpoint."
            )
            logger.warning(warn_msg)

    # Load scheduler state if provided and available
    if scheduler:
        if "scheduler_state_dict" in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                logger.debug("Loaded scheduler state.")
            except (RuntimeError, KeyError, ValueError, AttributeError) as e:
                err_msg = f"Failed to load scheduler_state_dict: {e}"
                logger.error(err_msg)
                # Non-fatal for backward compatibility
        else:
            logger.warning(
                "Scheduler provided, but 'scheduler_state_dict' not found "
                "in checkpoint."
            )

    # Return checkpoint data without model/optimizer/scheduler states
    checkpoint_data = {
        k: v
        for k, v in checkpoint.items()
        if k
        not in [
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
        ]
    }
    return checkpoint_data


def load_checkpoint_dict(
    checkpoint_path: str | Path, device: torch.device | None = None
) -> dict[str, Any]:
    """Load a checkpoint dictionary from file without loading weights into a
    model.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Optional device for map_location

    Returns:
        The loaded checkpoint dictionary
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device if device is not None else torch.device("cpu")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        logger.info(f"Loaded checkpoint dict from {checkpoint_path}")
        return checkpoint
    except (
        OSError,
        RuntimeError,
        pickle.UnpicklingError,
        zipfile.BadZipFile,
        KeyError,
        AttributeError,
    ) as e:  # More specific
        logger.error(
            f"Failed to load checkpoint dict from {checkpoint_path}: {e}"
        )
        raise


def adapt_legacy_checkpoint(
    legacy_checkpoint: dict[str, Any],
    training_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Adapt legacy checkpoint format to standardized format.

    Args:
        legacy_checkpoint: Legacy checkpoint dictionary
        training_config: Optional training configuration to add

    Returns:
        Adapted checkpoint with standardized metadata
    """
    adapted = legacy_checkpoint.copy()

    # Add missing required metadata
    if "pytorch_version" not in adapted:
        adapted["pytorch_version"] = torch.__version__

    if "timestamp" not in adapted:
        adapted["timestamp"] = datetime.now().isoformat()

    if training_config is not None and "config" not in adapted:
        adapted["config"] = training_config

    # Add platform info if missing
    if "python_version" not in adapted:
        adapted["python_version"] = (
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        )

    logger.info("Adapted legacy checkpoint to standardized format")
    return adapted


def load_and_adapt_legacy_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    device: torch.device | None = None,
    training_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load legacy checkpoint and adapt to standardized format.

    This function provides backward compatibility for older checkpoint formats.

    Args:
        checkpoint_path: Path to legacy checkpoint
        model: PyTorch model to load into
        optimizer: Optional optimizer
        scheduler: Optional scheduler
        device: Optional device
        training_config: Optional config to add to adapted checkpoint

    Returns:
        Adapted checkpoint data
    """
    # Load using standard function with no strict validation
    checkpoint_data = load_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        strict_validation=False,
    )

    # Adapt to standardized format
    return adapt_legacy_checkpoint(checkpoint_data, training_config)
