"""Checkpoint management utilities."""

# import os # Not needed if sorting by name
from pathlib import Path
from typing import Dict, Any, Optional, Union

import torch
from torch import nn
from torch.optim import Optimizer

from src.utils.logging import get_logger

logger = get_logger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    checkpoint_dir: Union[str, Path],
    filename: str = 'checkpoint.pt',
    additional_data: Optional[Dict[str, Any]] = None,
    keep_last_n: int = 1
) -> None:
    """Save a model checkpoint.

    Args:
        model: The PyTorch model to save
        optimizer: The optimizer to save
        epoch: Current training epoch
        checkpoint_dir: Directory to save checkpoints
        filename: Name of the checkpoint file. If using keep_last_n > 0,
                  this filename *must* follow a pattern where the epoch
                  number can be extracted, like 'prefix_epoch_NUM.pth'.
        additional_data: Optional dictionary of additional data to save
        keep_last_n: Number of recent checkpoints to keep (based on epoch
                     in filename)
    """
    # Ensure checkpoint_dir is a Path object
    checkpoint_dir = Path(checkpoint_dir)

    # Create directory if it doesn't exist
    try:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Checkpoint directory verified: {checkpoint_dir}")
    except (PermissionError, OSError) as e:
        logger.error(
            f"Failed to create checkpoint directory {checkpoint_dir}: {e}"
        )
        # Use a fallback directory
        fallback_dir = Path('./checkpoints')
        try:
            fallback_dir.mkdir(parents=True, exist_ok=True)
            logger.warning(
                f"Using fallback checkpoint directory: {fallback_dir}"
            )
            checkpoint_dir = fallback_dir
        except (PermissionError, OSError) as e:
            logger.error(
                f"Failed to create fallback directory {fallback_dir}: {e}"
            )
            raise RuntimeError(f"Cannot create checkpoint directory: {e}")

    # Prepare checkpoint path
    checkpoint_path = checkpoint_dir / filename

    # Verify that the path is writable by attempting to create a test file
    try:
        # Check if direct parent directory is writable
        test_file = checkpoint_dir / f"test_write_{epoch}.tmp"
        test_file.touch(exist_ok=True)
        test_file.unlink()  # Remove the test file
    except (PermissionError, OSError) as e:
        logger.error(f"Checkpoint directory is not writable: {e}")
        raise RuntimeError(f"Checkpoint directory is not writable: {e}")

    # Prepare checkpoint data
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    if additional_data:
        checkpoint.update(additional_data)

    # Save checkpoint with error handling
    try:
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint at epoch {epoch} to {checkpoint_path}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint to {checkpoint_path}: {e}")
        # Try an alternative path as fallback
        alt_path = checkpoint_dir / f"emergency_checkpoint_epoch_{epoch}.pt"
        try:
            torch.save(checkpoint, alt_path)
            logger.warning(f"Saved emergency checkpoint to {alt_path}")
        except Exception as e2:
            logger.error(f"Failed to save emergency checkpoint: {e2}")
            raise RuntimeError(f"Cannot save checkpoint: {e2}")

    # Clean up old checkpoints if needed, based on filename epoch number
    if keep_last_n > 0:
        # Define a common prefix/suffix pattern to parse epoch number
        # Example assumes: someprefix_epoch_123.pth
        # Adjust prefix/suffix if your filename pattern is different
        prefix = "_epoch_"
        suffix = ".pth"
        checkpoint_files = []
        glob_pattern = f'*{prefix}*{suffix}'

        for f_path in checkpoint_dir.glob(glob_pattern):
            try:
                # Extract epoch number from filename stem
                epoch_str = f_path.stem.split(prefix)[-1]
                file_epoch = int(epoch_str)
                checkpoint_files.append((file_epoch, f_path))
            except (IndexError, ValueError):
                log_msg = (
                    f"Could not parse epoch from {f_path.name}, "
                    f"skipping cleanup for this file."
                )
                logger.warning(log_msg)
                continue

        # Sort by epoch number, oldest first
        checkpoint_files.sort(key=lambda x: x[0])

        # Remove oldest checkpoints if count exceeds keep_last_n
        if len(checkpoint_files) > keep_last_n:
            files_to_remove = checkpoint_files[:-keep_last_n]
            logger.debug(f"Removing {len(files_to_remove)} old checkpoints")
            for file_epoch, f_path_to_remove in files_to_remove:
                try:
                    f_path_to_remove.unlink()
                    rm_msg = f"Removed old ckpt (epoch {file_epoch}): " \
                             f"{f_path_to_remove.name}"
                    logger.debug(rm_msg)
                except OSError as e:
                    err_msg = f"Error removing old checkpoint " \
                              f"{f_path_to_remove}: {e}"
                    logger.error(err_msg)


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
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
            map_location = torch.device('cpu')
            logger.debug("Model has no .device attribute, loading ckpt to CPU")

    # Load checkpoint
    try:
        # Set weights_only=True for security if only loading state_dicts
        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False  # Consider True if applicable
        )
        log_msg = f"Loading checkpoint from {checkpoint_path} to \
{map_location}"
        logger.info(log_msg)
    except Exception as e:
        logger.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")
        raise

    # Load model state
    if 'model_state_dict' not in checkpoint:
        raise KeyError(f"'model_state_dict' not found in {checkpoint_path}")
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        # Move model to target device *after* loading state dict
        if device:
            model.to(device)
    except Exception as e:
        logger.error(f"Failed to load model_state_dict: {e}")
        raise

    # Load optimizer state if provided and available
    if optimizer:
        if 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.debug("Loaded optimizer state.")
            except Exception as e:
                err_msg = f"Failed to load optimizer_state_dict: {e}"
                logger.error(err_msg)
                # Decide if this should be a fatal error or just a warning
                # raise # Consider re-raising or handling differently
        else:
            warn_msg = ("Optimizer provided, but 'optimizer_state_dict' "
                        "not found in checkpoint.")
            logger.warning(warn_msg)

    # Return checkpoint data without model/optimizer states
    checkpoint_data = {
        k: v for k, v in checkpoint.items()
        if k not in ['model_state_dict', 'optimizer_state_dict']
    }
    return checkpoint_data


def load_checkpoint_dict(
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
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

    map_location = device if device is not None else torch.device('cpu')
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False
        )
        logger.info(
            "Loaded checkpoint dict from %s to %s",
            str(checkpoint_path),
            str(map_location)
        )
        return checkpoint
    except Exception as e:
        logger.error(
            "Failed to load checkpoint dict from %s: %s",
            str(checkpoint_path),
            str(e)
        )
        raise
