"""Analysis module for SwinV2 hybrid experiments.

This module contains the generic experiment analyzer that works
with any dataset and image size for SwinV2 hybrid experiments.
"""

from .analyze_experiment import main as analyze_experiment

__all__ = [
    "analyze_experiment",
]
