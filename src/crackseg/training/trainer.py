"""Trainer: orchestrates training, validation, checkpointing, and early stopping.

This module provides the main Trainer class that coordinates all training operations
using modular components for better organization and maintainability.
"""

from omegaconf import DictConfig

from crackseg.utils import BaseLogger
from crackseg.utils.monitoring import BaseCallback
from crackseg.utils.training.early_stopping import EarlyStopping

from .components import (
    TrainerInitializer,
    TrainerSetup,
    TrainingLoop,
    ValidationLoop,
)
from .components.initializer import TrainingComponents


class Trainer:
    """Orchestrates the training and validation process using modular components."""

    def __init__(
        self,
        components: "TrainingComponents",
        cfg: DictConfig,
        logger_instance: BaseLogger | None = None,
        early_stopper: EarlyStopping | None = None,
        callbacks: list[BaseCallback] | None = None,
    ):
        """Initializes the Trainer with modular components."""
        # Initialize component instances
        self.initializer = TrainerInitializer()
        self.setup = TrainerSetup()
        self.training_loop = TrainingLoop()
        self.validation_loop = ValidationLoop()

        # Initialize core attributes
        self.initializer.initialize_core_attributes(
            self, components, cfg, logger_instance
        )
        self.initializer.parse_trainer_settings(self)

        # Setup components
        self.setup.setup_monitoring(self, callbacks)
        self.setup.setup_checkpointing_attributes(self)
        self.setup.setup_device_and_model(self)
        self.setup.setup_optimizer_and_scheduler(self)
        self.setup.setup_mixed_precision(self)
        self.setup.load_checkpoint_state(self)
        self.setup.setup_early_stopping_instance(self, early_stopper)
        self.setup.log_initialization_summary(self)

    def train(self) -> dict[str, float]:
        """Runs the full training loop using the training loop component."""
        return self.training_loop.train(self)

    def validate(self, epoch: int) -> dict[str, float]:
        """Runs validation using the validation loop component."""
        return self.validation_loop.validate(self, epoch)


__all__ = ["Trainer", "TrainingComponents"]
