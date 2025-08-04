"""Environment configuration for deployment.

This package provides environment configuration capabilities for different
deployment targets, including resource requirements, dependencies, and
deployment-specific settings.
"""

from .config import (
    ConfigurationResult,
    EnvironmentConfig,
    ResourceRequirements,
)
from .core import EnvironmentConfigurator

__all__ = [
    "EnvironmentConfigurator",
    "EnvironmentConfig",
    "ResourceRequirements",
    "ConfigurationResult",
]
