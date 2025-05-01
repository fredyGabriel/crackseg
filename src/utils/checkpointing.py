"""Utility functions for saving and loading model checkpoints."""

import torch
from torch.optim import Optimizer
from torch.amp import GradScaler
from pathlib import Path
from typing import Dict, Any, Optional
# from torch.optim.lr_scheduler import _LRScheduler # Add later if needed

from src.utils.loggers import get_logger

# Logger for this module
logger = get_logger(__name__)


def save_checkpoint(
    epoch: int,
    model: torch.nn.Module,
    optimizer: Optimizer,
    # scheduler: Optional[_LRScheduler] = None,
    scaler: Optional[GradScaler] = None,
    metrics: Optional[Dict[str, Any]] = None,
    checkpoint_dir: Path | str = "checkpoints",
    filename: str = "checkpoint_last.pth",
    is_best: bool = False,
    best_filename: str = "checkpoint_best.pth"
) -> None:
    """Saves the model checkpoint.

    Args:
        epoch: Current epoch number.
        model: Model instance.
        optimizer: Optimizer instance.
        scaler: GradScaler instance (if using AMP).
        metrics: Dictionary of metrics from validation (e.g., loss).
        checkpoint_dir: Directory to save the checkpoint.
        filename: Base filename for the checkpoint.
        is_best: Flag indicating if this is the best model so far.
        best_filename: Filename for the best model checkpoint.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # 'scheduler_state_dict': scheduler.state_dict() if scheduler else
        # None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'metrics': metrics if metrics else {}
    }

    filepath = checkpoint_dir / filename
    try:
        torch.save(state, filepath)
        logger.info(f"Checkpoint saved to {filepath}")

        if is_best:
            best_filepath = checkpoint_dir / best_filename
            torch.save(state, best_filepath)
            logger.info(f"Best checkpoint saved to {best_filepath}")

    except Exception as e:
        # Format error message to avoid long line
        logger.error(
            f"Error saving checkpoint to {filepath}: {e}", exc_info=True
        )


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,  # Using Any for flexibility
    scaler: Optional[GradScaler] = None,  # Add scaler parameter
    checkpoint_path: str = None,
    device: torch.device = None
) -> Dict[str, Any]:
    """Loads training state from a checkpoint file.

    Args:
        model: The model instance to load the state into.
        optimizer: The optimizer instance to load the state into (optional).
        scheduler: The scheduler instance to load the state into (optional).
        scaler: GradScaler instance to load state into (optional).
        checkpoint_path: Path to the checkpoint file.
        device: The device to map the loaded tensors to (optional).

    Returns:
        A dictionary containing the loaded state (epoch, best_metric, etc.),
        or an empty dictionary if checkpoint_path is None or file not found.
    """
    if not checkpoint_path or not Path(checkpoint_path).is_file():
        logger.warning(
            f"Checkpoint file not found at '{checkpoint_path}'. "
            f"Starting training from scratch."
        )
        # Default start state
        return {"epoch": 0, "best_metric_value": None}

    logger.info(f"Loading checkpoint from '{checkpoint_path}'")
    map_location = device if device else None
    # Explicitly set weights_only=False to load full state (optimizer, etc.)
    # and silence FutureWarning.
    checkpoint = torch.load(
        checkpoint_path, map_location=map_location, weights_only=False
    )

    # Load model state
    if 'model_state_dict' in checkpoint:
        # Handle potential DataParallel wrapping
        state_dict = checkpoint['model_state_dict']
        keys = list(state_dict.keys())
        is_dp_saved = keys[0].startswith('module.') if keys else False

        if isinstance(model, torch.nn.DataParallel):
            if not is_dp_saved:
                # Adjust state dict keys if saved model was not DataParallel
                state_dict = {'module.' + k: v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
        elif is_dp_saved:
            # Adjust state dict keys if saved model was DataParallel
            state_dict = {k.partition('module.')[2]: v for k, v in
                          state_dict.items()}
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
        logger.info("Model state loaded successfully.")
    else:
        logger.warning("Checkpoint missing 'model_state_dict'. "
                       "Model weights not loaded.")

    # Load optimizer state
    if optimizer and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("Optimizer state loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load optimizer state: {e}. "
                           "Optimizer state might be reset.")
    elif optimizer:
        logger.warning("Checkpoint missing 'optimizer_state_dict'. "
                       "Optimizer state not loaded.")

    # Load scheduler state
    if scheduler and 'scheduler_state_dict' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info("Scheduler state loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load scheduler state: {e}. "
                           "Scheduler state might be reset.")
    elif scheduler:
        logger.warning("Checkpoint missing 'scheduler_state_dict'. "
                       "Scheduler state not loaded.")

    # Load scaler state
    if scaler and 'scaler_state_dict' in checkpoint:
        try:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            logger.info("Scaler state loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load scaler state: {e}. Scaler state \
might be reset.")
    elif scaler:
        logger.warning("Checkpoint missing 'scaler_state_dict'. Scaler state \
not loaded.")

    # Load other metadata
    epoch = checkpoint.get("epoch", 0)
    best_metric_value = checkpoint.get("best_metric_value", None)
    logger.info(f"Checkpoint loaded. Resuming from epoch {epoch + 1}.")

    return {"epoch": epoch, "best_metric_value": best_metric_value}
