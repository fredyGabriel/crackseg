"""Compatibility layer for torchvision, timm, and albumentations imports.

This module provides safe imports and compatibility checking for optional
computer vision dependencies used in the CrackSeg project.
"""

from typing import Any

# Global availability flags
_timm_available = False
_albumentations_available = False
_torchvision_available = False

# Optional imports with availability tracking
try:
    import timm

    _timm_available = True
except ImportError:
    timm = None

try:
    import albumentations

    _albumentations_available = True
except ImportError:
    albumentations = None

try:
    import torchvision

    _torchvision_available = True
except ImportError:
    torchvision = None


def safe_import_timm() -> Any | None:
    """Safely import timm with error handling.

    Returns:
        timm module if available, None otherwise
    """
    if _timm_available:
        return timm  # type: ignore[return-value]
    return None


def safe_import_albumentations() -> Any | None:
    """Safely import albumentations with error handling.

    Returns:
        albumentations module if available, None otherwise
    """
    if _albumentations_available:
        return albumentations  # type: ignore[return-value]
    return None


def safe_import_torchvision() -> Any | None:
    """Safely import torchvision with error handling (legacy).

    Returns:
        torchvision module if available, None otherwise
    """
    if _torchvision_available:
        return torchvision  # type: ignore[return-value]
    return None


def is_timm_available() -> bool:
    """Check if timm is available.

    Returns:
        True if timm can be imported, False otherwise
    """
    return _timm_available


def is_albumentations_available() -> bool:
    """Check if albumentations is available.

    Returns:
        True if albumentations can be imported, False otherwise
    """
    return _albumentations_available


def is_torchvision_available() -> bool:
    """Check if torchvision is available (legacy).

    Returns:
        True if torchvision can be imported, False otherwise
    """
    return _torchvision_available


# Export commonly used components
if _timm_available:
    # Re-export commonly used timm functions
    pass

if _albumentations_available:
    # Re-export commonly used albumentations transforms
    pass

if _torchvision_available:
    # Re-export commonly used torchvision components
    pass
