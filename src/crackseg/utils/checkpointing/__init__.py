"""Checkpoint management utilities for CrackSeg project.

This module provides comprehensive checkpoint management capabilities including
saving, loading, validation, and legacy format support.
"""

from .config import (
    CheckpointLoadConfig,
    CheckpointSaveConfig,
    CheckpointSpec,
    generate_checkpoint_metadata,
)
from .legacy import (
    adapt_legacy_checkpoint,
    detect_legacy_format,
    get_legacy_checkpoint_info,
    load_and_adapt_legacy_checkpoint,
)
from .load import load_checkpoint, load_checkpoint_dict
from .save import create_standardized_filename, save_checkpoint
from .validation import (
    get_checkpoint_metadata,
    validate_checkpoint_completeness,
    validate_checkpoint_format,
    verify_checkpoint_integrity,
)

__all__ = [
    # Configuration
    "CheckpointSpec",
    "CheckpointSaveConfig",
    "CheckpointLoadConfig",
    "generate_checkpoint_metadata",
    # Core functionality
    "save_checkpoint",
    "load_checkpoint",
    "load_checkpoint_dict",
    "create_standardized_filename",
    # Validation
    "verify_checkpoint_integrity",
    "validate_checkpoint_completeness",
    "validate_checkpoint_format",
    "get_checkpoint_metadata",
    # Legacy support
    "adapt_legacy_checkpoint",
    "load_and_adapt_legacy_checkpoint",
    "detect_legacy_format",
    "get_legacy_checkpoint_info",
]
