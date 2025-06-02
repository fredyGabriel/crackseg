"""Experiment management utilities for the Crack Segmentation project.

This module provides experiment setup, management, and coordination utilities.
"""

from .experiment import initialize_experiment
from .manager import ExperimentManager

__all__ = [
    # Experiment setup
    "initialize_experiment",
    # Experiment management
    "ExperimentManager",
]
