"""Visualization templates system.

This package provides configurable templates for consistent
visualization styling across different plot types.
"""

from .base_template import BaseVisualizationTemplate
from .prediction_template import PredictionVisualizationTemplate
from .training_template import TrainingVisualizationTemplate

__all__ = [
    "BaseVisualizationTemplate",
    "TrainingVisualizationTemplate",
    "PredictionVisualizationTemplate",
]
