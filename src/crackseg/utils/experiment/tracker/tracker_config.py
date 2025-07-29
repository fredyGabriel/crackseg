"""
Configuration handling for ExperimentTracker.

This module provides configuration handling methods for the ExperimentTracker
component.
"""

import hashlib
from typing import Any

from omegaconf import DictConfig, OmegaConf


class ExperimentConfigManager:
    """Manages configuration handling for experiments."""

    @staticmethod
    def calculate_config_hash(config: DictConfig) -> str:
        """Calculate hash of configuration for reproducibility."""
        config_str = OmegaConf.to_yaml(config, resolve=True)
        return hashlib.sha256(config_str.encode()).hexdigest()

    @staticmethod
    def extract_config_summary(config: DictConfig) -> dict[str, Any]:
        """Extract key configuration parameters for summary."""
        config_dict = OmegaConf.to_container(config, resolve=True)

        summary = {}

        # Extract key training parameters
        if isinstance(config_dict, dict) and "training" in config_dict:
            training = config_dict["training"]
            if isinstance(training, dict):
                summary["epochs"] = training.get("epochs", "unknown")
                summary["batch_size"] = training.get("batch_size", "unknown")
                summary["learning_rate"] = training.get(
                    "learning_rate", "unknown"
                )
                summary["optimizer"] = training.get("optimizer", {}).get(
                    "_target_", "unknown"
                )
                summary["loss"] = training.get("loss", {}).get(
                    "_target_", "unknown"
                )

        # Extract model parameters
        if isinstance(config_dict, dict) and "model" in config_dict:
            model = config_dict["model"]
            if isinstance(model, dict):
                summary["model_type"] = model.get("_target_", "unknown")
                summary["encoder"] = model.get("encoder", {}).get(
                    "_target_", "unknown"
                )
                summary["decoder"] = model.get("decoder", {}).get(
                    "_target_", "unknown"
                )

        # Extract data parameters
        if isinstance(config_dict, dict) and "data" in config_dict:
            data = config_dict["data"]
            if isinstance(data, dict):
                summary["dataset"] = data.get("dataset", "unknown")
                summary["data_root"] = data.get("root_dir", "unknown")

        return summary
