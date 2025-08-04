"""Trainer initialization component.

Handles the initialization of core trainer attributes and configuration validation.
"""

from dataclasses import dataclass
from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from crackseg.training.config_validation import validate_trainer_config
from crackseg.utils import BaseLogger
from crackseg.utils.logging.setup import setup_internal_logger


@dataclass
class TrainingComponents:
    """Encapsulates the core components required for training."""

    model: torch.nn.Module
    train_loader: DataLoader[Any]
    val_loader: DataLoader[Any]
    loss_fn: torch.nn.Module
    metrics_dict: dict[str, Any]


class TrainerInitializer:
    """Handles trainer initialization and core attribute setup."""

    def __init__(self) -> None:
        """Initialize the trainer initializer."""
        pass

    def initialize_core_attributes(
        self,
        trainer_instance: Any,
        components: TrainingComponents,
        cfg: DictConfig,
        logger_instance: BaseLogger | None,
    ) -> None:
        """Validates config and initializes core trainer attributes."""
        # Handle experiment namespace configuration access for training
        training_cfg = self._extract_training_config(cfg)
        validate_trainer_config(training_cfg)

        # Set core attributes
        trainer_instance.full_cfg = cfg
        trainer_instance.cfg = training_cfg  # Main trainer config node
        trainer_instance.model = components.model
        trainer_instance.train_loader = components.train_loader
        trainer_instance.val_loader = components.val_loader
        trainer_instance.loss_fn = components.loss_fn
        trainer_instance.metrics_dict = components.metrics_dict
        trainer_instance.logger_instance = logger_instance
        trainer_instance.internal_logger = setup_internal_logger(
            logger_instance
        )
        trainer_instance.grad_accum_steps = training_cfg.get(
            "gradient_accumulation_steps", 1
        )
        trainer_instance.verbose = training_cfg.get("verbose", True)
        trainer_instance.start_epoch = 1  # Default start epoch

    def _extract_training_config(self, cfg: DictConfig) -> DictConfig:
        """Extract training configuration from the full config."""
        training_cfg = None

        # Look in experiments namespace for training config
        if "experiments" in cfg:
            for exp_name in cfg.experiments:
                exp_config = cfg.experiments[exp_name]
                if hasattr(exp_config, "training"):
                    training_cfg = exp_config.training
                    break

        # Fall back to direct access
        if training_cfg is None and hasattr(cfg, "training"):
            training_cfg = cfg.training

        if training_cfg is None:
            raise ValueError("No training configuration found in config")

        return training_cfg

    def parse_trainer_settings(self, trainer_instance: Any) -> None:
        """Parses basic trainer settings from the configuration."""
        trainer_instance.epochs = trainer_instance.cfg.get("epochs", 10)
        trainer_instance.device_str = trainer_instance.cfg.get(
            "device", "auto"
        )
        trainer_instance.use_amp = trainer_instance.cfg.get("use_amp", True)
        trainer_instance.verbose = trainer_instance.cfg.get("verbose", True)
        trainer_instance.start_epoch = 1  # Default start epoch
