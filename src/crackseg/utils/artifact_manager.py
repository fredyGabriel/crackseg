"""
Artifact Manager for CrackSeg project.

This module provides comprehensive artifact management functionality for ML
experiments, including model saving, metric tracking, visualization storage,
and integrity validation.

This is the main entry point that provides backward compatibility with the
original ArtifactManager interface.
"""

from .artifact_manager.core import ArtifactManager, ArtifactManagerConfig
from .artifact_manager.metadata import ArtifactMetadata

# Re-export for backward compatibility
__all__ = [
    "ArtifactManager",
    "ArtifactManagerConfig",
    "ArtifactMetadata",
]
