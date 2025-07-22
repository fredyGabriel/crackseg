"""
Configuration data factory for E2E testing. This module provides
configuration file generation for test scenarios.
"""

import tempfile
from pathlib import Path
from typing import Any

import yaml

from .base import BaseDataFactory, TestData


class ConfigDataFactory(BaseDataFactory):
    """Factory for generating test configuration files."""

    def generate(
        self,
        config_type: str = "basic",
        invalid: bool = False,
        missing_keys: list[str] | None = None,
        **kwargs: Any,
    ) -> TestData:
        """
        Generate test configuration data. Args: config_type: Type of config to
        generate ('basic', 'advanced', 'model', 'training') invalid: Whether
        to generate invalid configuration missing_keys: List of keys to omit
        for invalid configs **kwargs: Additional configuration parameters
        Returns: TestData containing generated configuration
        """
        # Configuration templates
        config_templates = {
            "basic": {
                "model": {
                    "name": "test_model",
                    "encoder": "resnet50",
                    "decoder": "unet",
                    "num_classes": 2,
                },
                "training": {
                    "batch_size": 4,
                    "epochs": 1,
                    "learning_rate": 0.001,
                },
                "data": {"input_size": [512, 512], "num_channels": 3},
            },
            "advanced": {
                "model": {
                    "name": "advanced_model",
                    "encoder": "swin_transformer",
                    "decoder": "fpn",
                    "num_classes": 2,
                    "pretrained": True,
                },
                "training": {
                    "batch_size": 8,
                    "epochs": 5,
                    "learning_rate": 0.0001,
                    "optimizer": "adamw",
                    "scheduler": "cosine",
                },
                "data": {
                    "input_size": [256, 256],
                    "num_channels": 3,
                    "augmentation": True,
                },
            },
        }

        config = config_templates.get(
            config_type, config_templates["basic"]
        ).copy()

        config.update(kwargs)

        # Make invalid if requested
        if invalid:
            if missing_keys:
                for key in missing_keys:
                    if key in config:
                        del config[key]
            else:
                config["model"]["num_classes"] = -1
                config["training"]["batch_size"] = 0

        # Generate temporary file
        temp_dir = (
            self.environment_manager.state["artifacts_dir"]
            if self.environment_manager
            else Path(tempfile.gettempdir())
        )
        temp_dir.mkdir(exist_ok=True)

        config_file = temp_dir / f"test_config_{config_type}_{id(config)}.yaml"

        with open(config_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        if self.environment_manager:
            self.environment_manager.register_temp_file(config_file)

        return {
            "data_type": "config",
            "file_path": config_file,
            "metadata": {
                "config_type": config_type,
                "invalid": invalid,
                "format": "yaml",
            },
            "cleanup_required": True,
        }
