"""Core utilities for the Crack Segmentation project.

This module provides fundamental utilities used throughout the application
including device handling, exception classes, path utilities, and seed
management.
"""

from .device import get_device
from .exceptions import (
    ConfigError,
    CrackSegError,
    DataError,
    EvaluationError,
    ModelError,
    ResourceError,
    TrainingError,
    ValidationError,
)
from .paths import ensure_dir, get_abs_path
from .seeds import set_random_seeds

__all__ = [
    # Device utilities
    "get_device",
    # Exception classes
    "CrackSegError",
    "ConfigError",
    "DataError",
    "EvaluationError",
    "ModelError",
    "ResourceError",
    "TrainingError",
    "ValidationError",
    # Path utilities
    "ensure_dir",
    "get_abs_path",
    # Seed management
    "set_random_seeds",
]
