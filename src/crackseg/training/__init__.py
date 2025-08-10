"""Training helpers under the `crackseg.training` namespace.

This package consolidates training pipeline utilities and exposes the public
API for training components and the main `Trainer`.
"""

from __future__ import annotations

from .checkpoint_manager import handle_checkpointing_and_resume  # noqa: F401
from .components import (  # noqa: F401
    TrainerInitializer,
    TrainerSetup,
    TrainingLoop,
    ValidationLoop,
)
from .data_loading import load_data  # noqa: F401

# Re-export helpers (kept for public API)
from .environment_setup import setup_environment  # noqa: F401
from .model_creation import create_model  # noqa: F401

# Re-export core training classes
from .trainer import Trainer, TrainingComponents  # noqa: F401
from .training_setup import setup_training_components  # noqa: F401

__all__ = [
    # helpers
    "setup_environment",
    "load_data",
    "create_model",
    "setup_training_components",
    "handle_checkpointing_and_resume",
    # trainer API
    "Trainer",
    "TrainingComponents",
    "TrainerInitializer",
    "TrainerSetup",
    "TrainingLoop",
    "ValidationLoop",
]
