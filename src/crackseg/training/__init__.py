"""Training module for crack segmentation models.

This module provides the core training infrastructure including the main Trainer
class and supporting components for model training, validation, and experiment
management.
"""

from .components import (
    TrainerInitializer,
    TrainerSetup,
    TrainingLoop,
    ValidationLoop,
)
from .trainer import Trainer, TrainingComponents

__all__ = [
    "Trainer",
    "TrainingComponents",
    "TrainerInitializer",
    "TrainerSetup",
    "TrainingLoop",
    "ValidationLoop",
]
