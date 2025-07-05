"""Base classes and utilities for GUI workflow integration testing.

This module provides common base classes and utilities used across
all workflow integration tests for the CrackSeg GUI application.
"""

import tempfile
from pathlib import Path
from typing import Any

import yaml


class WorkflowTestBase:
    """Base class for workflow testing with common utilities."""

    def setup_method(self) -> None:
        """Setup method for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def teardown_method(self) -> None:
        """Cleanup after each test."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_valid_config_file(
        self, content: dict[str, Any] | None = None
    ) -> Path:
        """Create a valid configuration file for testing.

        Args:
            content: Configuration content or use default

        Returns:
            Path to created config file
        """
        if content is None:
            content = {
                "defaults": [
                    "data: default",
                    "model: default",
                    "training: default",
                ],
                "experiment": {"name": "test_experiment", "random_seed": 42},
                "model": {"name": "unet", "encoder": "resnet50"},
                "training": {"epochs": 10, "learning_rate": 0.001},
                "data": {"batch_size": 4, "image_size": [512, 512]},
            }

        config_file = self.temp_path / "test_config.yaml"
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(content, f, default_flow_style=False)

        return config_file

    def create_invalid_config_file(self) -> Path:
        """Create an invalid configuration file for error testing.

        Returns:
            Path to created invalid config file
        """
        invalid_config = self.temp_path / "invalid_config.yaml"
        with open(invalid_config, "w", encoding="utf-8") as f:
            f.write("invalid_yaml: [unclosed list\nmodel: test")
        return invalid_config
