"""Handles model checkpoint saving and loading."""

import os
import shutil
import torch
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool,
    checkpoint_dir: str,
    filename: str = "checkpoint.pth.tar",
    best_filename: str = "model_best.pth.tar"
) -> None:
    """Saves the training state to a checkpoint file.

    Args:
        state: Dictionary containing model state_dict, optimizer state_dict,
               epoch, best_metric_value, etc.
        is_best: Boolean flag indicating if this checkpoint is the best so far.
        checkpoint_dir: Directory where checkpoints will be saved.
        filename: The name of the checkpoint file for the current state.
        best_filename: The name of the checkpoint file for the best state.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    logger.info(f"Checkpoint saved to {filepath} (Epoch {state.get('epoch')})")

    if is_best:
        best_filepath = os.path.join(checkpoint_dir, best_filename)
        shutil.copyfile(filepath, best_filepath)
        logger.info(f"Best model checkpoint updated to {best_filepath}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,  # Using Any for flexibility
    checkpoint_path: str = None,
    device: torch.device = None
) -> Dict[str, Any]:
    """Loads training state from a checkpoint file.

    Args:
        model: The model instance to load the state into.
        optimizer: The optimizer instance to load the state into (optional).
        scheduler: The scheduler instance to load the state into (optional).
        checkpoint_path: Path to the checkpoint file.
        device: The device to map the loaded tensors to (optional).

    Returns:
        A dictionary containing the loaded state (epoch, best_metric, etc.),
        or an empty dictionary if checkpoint_path is None or file not found.
    """
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        logger.warning(
            f"Checkpoint file not found at '{checkpoint_path}'. "
            f"Starting training from scratch."
        )
        # Default start state
        return {"epoch": 0, "best_metric_value": None}

    logger.info(f"Loading checkpoint from '{checkpoint_path}'")
    map_location = device if device else None
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

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

    # Load other metadata
    epoch = checkpoint.get("epoch", 0)
    best_metric_value = checkpoint.get("best_metric_value", None)
    logger.info(f"Checkpoint loaded. Resuming from epoch {epoch + 1}.")

    return {"epoch": epoch, "best_metric_value": best_metric_value}
