"""Experiment management utilities for the Crack Segmentation project.

This module provides experiment setup, management, and coordination utilities.
"""

from .experiment import initialize_experiment
from .manager import ExperimentManager
from .metadata import ExperimentMetadata

# Import ExperimentTracker from the main tracker file
try:
    from .tracker import ExperimentTracker
except ImportError:
    # Try importing from the specific file to avoid conflict with subpackage
    import importlib.util
    import os

    spec = importlib.util.spec_from_file_location(
        "tracker", os.path.join(os.path.dirname(__file__), "tracker.py")
    )
    if spec and spec.loader:
        tracker_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tracker_module)
        ExperimentTracker = getattr(tracker_module, "ExperimentTracker", None)
    else:
        ExperimentTracker = None

__all__ = [
    # Experiment setup
    "initialize_experiment",
    # Experiment management
    "ExperimentManager",
    # Experiment tracking
    "ExperimentTracker",
    "ExperimentMetadata",
]
