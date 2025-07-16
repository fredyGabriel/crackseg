"""Checkpointing utilities for the Crack Segmentation project.

This module provides comprehensive checkpoint management including saving,
loading, validation, and helper functions for training loops.
"""

from .core import (
    CheckpointSaveConfig,
    CheckpointSpec,
    adapt_legacy_checkpoint,
    create_standardized_filename,
    generate_checkpoint_metadata,
    load_checkpoint,
    load_checkpoint_dict,
    save_checkpoint,
    validate_checkpoint_completeness,
    verify_checkpoint_integrity,
)
from .helpers import (
    CheckpointConfig,
    CheckpointContext,
    handle_epoch_checkpointing,
)
from .setup import setup_checkpointing

__all__ = [
    # Core checkpointing
    "CheckpointSaveConfig",
    "CheckpointSpec",
    "save_checkpoint",
    "load_checkpoint",
    "load_checkpoint_dict",
    "verify_checkpoint_integrity",
    # Additional core functions
    "adapt_legacy_checkpoint",
    "create_standardized_filename",
    "generate_checkpoint_metadata",
    "validate_checkpoint_completeness",
    # Helper functions
    "CheckpointConfig",
    "CheckpointContext",
    "handle_epoch_checkpointing",
    # Setup
    "setup_checkpointing",
]
