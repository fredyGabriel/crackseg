"""
ExperimentTracker submodules.

This package contains the modular components of the ExperimentTracker:
- lifecycle: Experiment lifecycle management
- artifacts: Artifact association and retrieval
- config: Configuration handling and hashing
- git: Git metadata collection
"""

from .tracker_artifacts import ExperimentArtifactManager
from .tracker_config import ExperimentConfigManager
from .tracker_git import ExperimentGitManager
from .tracker_lifecycle import ExperimentLifecycleManager

__all__ = [
    "ExperimentArtifactManager",
    "ExperimentConfigManager",
    "ExperimentGitManager",
    "ExperimentLifecycleManager",
]
