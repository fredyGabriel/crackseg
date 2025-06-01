"""
Clean loss registry implementation.
This module provides a registration system that avoids circular dependencies
by using lazy loading and clear separation of concerns.
"""

from .clean_registry import CleanLossRegistry
from .enhanced_registry import (
    EnhancedLossRegistry,
    ParameterValidationError,
    TypeValidationError,
)
from .setup_losses import setup_standard_losses

# Global registry instance (using enhanced registry for production)
registry = EnhancedLossRegistry()

# Auto-setup standard losses on import
setup_standard_losses(registry)

__all__ = [
    "registry",
    "CleanLossRegistry",
    "EnhancedLossRegistry",
    "ParameterValidationError",
    "TypeValidationError",
    "setup_standard_losses",
]
