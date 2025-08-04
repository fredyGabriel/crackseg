"""Legacy checkpoint handling functionality.

This module handles backward compatibility for older checkpoint formats
and provides adaptation utilities.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from .load import load_checkpoint

logger = logging.getLogger(__name__)


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


def detect_legacy_format(checkpoint_data: dict[str, Any]) -> bool:
    """Detect if checkpoint is in legacy format.

    Args:
        checkpoint_data: Checkpoint dictionary to analyze

    Returns:
        True if checkpoint appears to be in legacy format
    """
    # Legacy checkpoints typically lack modern metadata
    modern_fields = {"pytorch_version", "python_version", "timestamp"}

    missing_modern_fields = modern_fields - set(checkpoint_data.keys())

    # If more than half of modern fields are missing, likely legacy
    return len(missing_modern_fields) > len(modern_fields) / 2


def get_legacy_checkpoint_info(
    checkpoint_data: dict[str, Any],
) -> dict[str, Any]:
    """Extract information from legacy checkpoint format.

    Args:
        checkpoint_data: Legacy checkpoint dictionary

    Returns:
        Dictionary with legacy checkpoint information
    """
    info = {
        "is_legacy": True,
        "has_model_state": "model_state_dict" in checkpoint_data,
        "has_optimizer_state": "optimizer_state_dict" in checkpoint_data,
        "has_scheduler_state": "scheduler_state_dict" in checkpoint_data,
        "epoch": checkpoint_data.get("epoch"),
        "best_metric": checkpoint_data.get("best_metric_value"),
    }

    # Try to extract any available metadata
    if "pytorch_version" in checkpoint_data:
        info["pytorch_version"] = checkpoint_data["pytorch_version"]
    if "timestamp" in checkpoint_data:
        info["timestamp"] = checkpoint_data["timestamp"]

    return info
