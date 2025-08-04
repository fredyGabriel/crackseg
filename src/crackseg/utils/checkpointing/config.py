"""Configuration classes for checkpoint management.

This module contains the data classes and configuration objects used
by the checkpoint management system, providing validation and type safety.
"""

import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


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


@dataclass
class CheckpointLoadConfig:
    """Configuration for loading a checkpoint."""

    strict_validation: bool = False
    device: torch.device | None = None
    map_location: str | torch.device | None = None


def generate_checkpoint_metadata() -> dict[str, Any]:
    """Generate automatic metadata for checkpoint.

    Returns:
        Dictionary containing automatic metadata for checkpoints
    """
    return {
        "pytorch_version": torch.__version__,
        "python_version": (
            f"{sys.version_info.major}.{sys.version_info.minor}."
            f"{sys.version_info.micro}"
        ),
        "platform": platform.platform(),
        "timestamp": datetime.now().isoformat(),
    }
