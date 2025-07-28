"""Model loading utilities for crack segmentation evaluation."""

import importlib
import logging
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles loading of trained models and configurations."""

    def __init__(self, checkpoint_path: str | Path) -> None:
        """
        Initialize the model loader.

        Args:
            checkpoint_path: Path to the model checkpoint
        """
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    def load_checkpoint(self) -> dict[str, Any]:
        """Load checkpoint with PyTorch 2.6+ compatibility."""
        try:
            # Try with weights_only=False for full checkpoint
            checkpoint = torch.load(
                self.checkpoint_path, map_location="cpu", weights_only=False
            )
        except Exception as e:
            logger.warning(f"Failed to load full checkpoint: {e}")
            # Fallback to weights_only=True
            checkpoint = torch.load(
                self.checkpoint_path, map_location="cpu", weights_only=True
            )

        return checkpoint

    def load_config(
        self, config_path: str | Path | None = None
    ) -> DictConfig | ListConfig:
        """
        Load model configuration.

        Args:
            config_path: Optional path to config file

        Returns:
            Model configuration
        """
        checkpoint = self.load_checkpoint()

        # Try config_path first
        if config_path and Path(config_path).exists():
            config = OmegaConf.load(config_path)
        elif "config" in checkpoint:
            config = OmegaConf.create(checkpoint["config"])
        else:
            raise ValueError(
                "No configuration found in checkpoint or config_path"
            )

        return config

    def create_model(self, config: DictConfig) -> torch.nn.Module:
        """
        Create model from configuration.

        Args:
            config: Model configuration

        Returns:
            Instantiated PyTorch model
        """
        model_config = config.model

        if not hasattr(model_config, "_target_"):
            raise ValueError("Model configuration must have _target_ field")

        # Import model class dynamically
        module_path, class_name = model_config._target_.rsplit(".", 1)
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)

        # Filter out _target_ from config
        model_kwargs = {
            k: v for k, v in model_config.items() if k != "_target_"
        }

        model = model_class(**model_kwargs)

        # Load weights if available
        checkpoint = self.load_checkpoint()
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(
                f"Loaded model weights from epoch "
                f"{checkpoint.get('epoch', 'unknown')}"
            )
        else:
            raise ValueError("No model_state_dict found in checkpoint")

        return model

    def get_model_info(self, config: DictConfig) -> dict[str, Any]:
        """
        Get information about the model.

        Args:
            config: Model configuration

        Returns:
            Dictionary with model information
        """
        model = self.create_model(config)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        return {
            "model_type": config.model._target_,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "checkpoint_path": str(self.checkpoint_path),
            "input_size": config.data.image_size,
            "num_classes": config.model.num_classes,
        }
