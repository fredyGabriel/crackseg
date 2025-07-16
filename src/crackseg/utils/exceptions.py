"""
Compatibility module for exception imports.

This module re-exports exceptions from crackseg.utils.core.exceptions to maintain
backward compatibility with existing imports.
"""

# Re-export all exceptions from core module
from crackseg.utils.core.exceptions import (
    ConfigError,
    CrackSegError,
    DataError,
    EvaluationError,
    ModelError,
    ResourceError,
    TrainingError,
    ValidationError,
)

__all__ = [
    "CrackSegError",
    "ConfigError",
    "DataError",
    "ModelError",
    "ResourceError",
    "TrainingError",
    "ValidationError",
    "EvaluationError",
]
