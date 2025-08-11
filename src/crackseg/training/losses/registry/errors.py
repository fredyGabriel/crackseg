"""Registry-specific error types for loss registry modules.

These exceptions are kept in a dedicated module to keep registry
implementations focused and within line limits.
"""

from __future__ import annotations

from .clean_registry import RegistryError


class ParameterValidationError(RegistryError):
    """Raised when loss parameters fail validation."""

    pass


class TypeValidationError(RegistryError):
    """Raised when instantiated loss doesn't match expected type."""

    pass
