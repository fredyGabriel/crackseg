"""
Artifact Manager for CrackSeg project.

This module provides comprehensive artifact management functionality for ML
experiments, including model saving, metric tracking, visualization storage,
and integrity validation.

This is the main entry point that provides backward compatibility with the
original ArtifactManager interface.
"""

from .core import ArtifactManager, ArtifactManagerConfig
from .metadata import ArtifactMetadata
from .versioning import ArtifactVersion, ArtifactVersioner, VersionInfo

# Re-export for backward compatibility
__all__ = [
    "ArtifactManager",
    "ArtifactManagerConfig",
    "ArtifactMetadata",
    "ArtifactVersioner",
    "VersionInfo",
    "ArtifactVersion",
]
