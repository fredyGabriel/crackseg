"""Packaging system for deployment artifacts.

This package provides automated packaging capabilities including
containerization, dependency management, and environment isolation.
"""

from .config import ContainerizationConfig, PackagingResult
from .core import PackagingSystem

__all__ = [
    "PackagingSystem",
    "PackagingResult",
    "ContainerizationConfig",
]
