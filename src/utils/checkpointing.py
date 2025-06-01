"""Checkpoint management utilities."""

import pickle  # Import pickle for specific exceptions
import zipfile  # Import zipfile for specific exceptions
from dataclasses import dataclass  # Import dataclass
from logging import Logger
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CheckpointSaveConfig:
    """Configuration for saving a checkpoint."""

    checkpoint_dir: str | Path
    filename: str = "checkpoint.pt"
    additional_data: dict[str, Any] | None = None
    keep_last_n: int = 1


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
        torch.save(checkpoint, checkpoint_path)  # type: ignore[reportUnknownMemberType]
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
            torch.save(checkpoint, alt_path)  # type: ignore[reportUnknownMemberType]
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
) -> None:
    """Save a model checkpoint."""
    checkpoint_dir = _ensure_checkpoint_dir(
        Path(config.checkpoint_dir), logger
    )
    checkpoint_path = checkpoint_dir / config.filename

    checkpoint_data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if config.additional_data:
        checkpoint_data.update(config.additional_data)

    _save_to_path(checkpoint_data, checkpoint_path, epoch, logger)
    _cleanup_old_checkpoints(checkpoint_dir, config.keep_last_n, logger)


@dataclass
class CheckpointLoadConfig:  # Renamed for clarity if needed, or keep as is
    """Configuration for loading a checkpoint (Not used in save_checkpoint)."""

    # Parameters relevant to loading, if any specific ones arise.
    # For now, load_checkpoint and load_checkpoint_dict manage their own args.
    pass


# load_checkpoint and load_checkpoint_dict remain largely unchanged for now,
# as their argument counts are acceptable.
# Consider refactoring them similarly if their complexity grows.


def load_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Load a model checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        model: The PyTorch model to load weights into
        optimizer: Optional optimizer to load state into
        device: Optional device to load the model to

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
        checkpoint = torch.load(  # type: ignore[reportUnknownMemberType]
            checkpoint_path,
            map_location=map_location,
            weights_only=False,  # Consider True if applicable
        )
        log_msg = (
            f"Loading checkpoint from {checkpoint_path} to {map_location}"
        )
        logger.info(log_msg)
    except Exception as e:
        # torch.load can raise several types of exceptions depending on the
        # error, from I/O problems to deserialization errors
        logger.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")
        raise

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

    # Return checkpoint data without model/optimizer states
    checkpoint_data = {
        k: v
        for k, v in checkpoint.items()
        if k not in ["model_state_dict", "optimizer_state_dict"]
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

    map_location = device if device is not None else torch.device("cpu")
    try:
        checkpoint = torch.load(  # type: ignore[reportUnknownMemberType]
            checkpoint_path,
            map_location=map_location,
            weights_only=False,  # Ensure this is intended
        )
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
