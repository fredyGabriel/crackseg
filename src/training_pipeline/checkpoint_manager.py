"""Checkpoint management for training pipeline.

This module provides checkpoint handling functions for the training pipeline
including checkpoint loading, saving, and resume functionality.
"""

import logging
from typing import Any

from omegaconf import DictConfig
from torch import optim

from crackseg.utils import load_checkpoint

# Configure standard logger
log = logging.getLogger(__name__)


def handle_checkpointing_and_resume(
    cfg: DictConfig,
    model: Any,
    optimizer: optim.Optimizer,
    device: Any,
    experiment_logger: Any,
) -> tuple[int, float | None]:
    """
    Handle checkpoint loading and resume functionality for training.

    This function manages the checkpoint system including:
    - Loading checkpoints from specified paths
    - Resuming training from saved states
    - Handling checkpoint validation and error recovery
    - Logging checkpoint information

    Args:
        cfg: Hydra configuration containing checkpoint settings.
            Expected structure:
            - cfg.training.checkpoints.resume_from_checkpoint (str, optional):
                Path to checkpoint file to resume from
            - cfg.training.checkpoints.save_freq (int, optional): Frequency
                of checkpoint saving
            - cfg.training.checkpoints.save_best (DictConfig, optional):
                Best model saving configuration

        model: The neural network model to load state into.

        optimizer: The optimizer to load state into.

        device: Target device for model placement.

        experiment_logger: Logger instance for experiment tracking.

    Returns:
        tuple[int, float | None]: A tuple containing:
            - start_epoch: The epoch to start training from
            - best_metric_value: The best metric value from checkpoint

    Examples:
        ```python
        cfg = OmegaConf.create({
            "training": {
                "checkpoints": {
                    "resume_from_checkpoint": "checkpoints/model.pth",
                    "save_freq": 10,
                    "save_best": {
                        "enabled": True,
                        "monitor_metric": "val_iou"
                    }
                }
            }
        })

        start_epoch, best_metric = handle_checkpointing_and_resume(
            cfg, model, optimizer, device, logger
        )

        print(f"Starting from epoch: {start_epoch}")
        print(f"Best metric: {best_metric}")
        ```

    Note:
        The function provides comprehensive error handling for checkpoint
        loading and validates checkpoint compatibility with current model.
    """
    log.info("Handling checkpointing and resume...")

    # Initialize default values
    start_epoch = 0
    best_metric_value = None

    # Check if we should resume from checkpoint
    # Handle experiment namespace configuration access
    training_cfg = None
    if "experiments" in cfg:
        # Look in experiments namespace for training config
        for exp_name in cfg.experiments:
            exp_config = cfg.experiments[exp_name]
            if hasattr(exp_config, "training"):
                training_cfg = exp_config.training
                break

    # Fall back to direct access
    if training_cfg is None and hasattr(cfg, "training"):
        training_cfg = cfg.training

    checkpoint_path = (
        getattr(training_cfg, "checkpoint_dir", None) if training_cfg else None
    )
    if checkpoint_path:
        log.info("Resuming from checkpoint: %s", checkpoint_path)
        try:
            checkpoint = load_checkpoint(checkpoint_path, device)

            # Load model state
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                log.info("Model state loaded from checkpoint")

            # Load optimizer state
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                log.info("Optimizer state loaded from checkpoint")

            # Get training state
            if "epoch" in checkpoint:
                start_epoch = checkpoint["epoch"] + 1
                log.info("Resuming from epoch: %s", start_epoch)

            if "best_metric_value" in checkpoint:
                best_metric_value = checkpoint["best_metric_value"]
                log.info(
                    "Best metric value from checkpoint: %s", best_metric_value
                )

            log.info("Checkpoint loaded successfully")

        except Exception as e:
            log.error("Error loading checkpoint: %s", str(e))
            log.warning("Starting training from scratch")
            start_epoch = 0
            best_metric_value = None
    else:
        log.info("No checkpoint specified, starting from scratch")

    return start_epoch, best_metric_value
