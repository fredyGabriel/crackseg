"""Checkpoint loading functionality.

This module handles the loading of model checkpoints with enhanced
validation and compatibility.
"""

import logging
import pickle
import zipfile
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from .config import CheckpointSpec

logger = logging.getLogger(__name__)


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

    device = device if device is not None else torch.device("cpu")
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
