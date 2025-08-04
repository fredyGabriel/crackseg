"""
Test data automation component for workflow automation scenarios. This
module provides automated test data generation capabilities for
supporting the workflow automation framework with realistic test
scenarios.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml


class TestDataAutomator:
    """Automated test data generation for workflow automation scenarios."""

    def __init__(self, base_path: Path) -> None:
        """
        Initialize test data automator with base path for generated files.
        Args: base_path: Base directory for generated test data files
        """
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)

    def generate_configuration_scenarios(self) -> list[dict[str, Any]]:
        """
        Generate various configuration scenarios for automated testing.
        Returns: List of configuration scenarios with metadata
        """
        scenarios = []

        # Valid baseline configuration
        scenarios.append(
            {
                "name": "valid_baseline",
                "description": (
                    "Standard valid configuration for crack segmentation"
                ),
                "config": {
                    "model": {
                        "name": "unet",
                        "encoder": "resnet50",
                        "decoder": "unet",
                        "num_classes": 2,
                    },
                    "training": {
                        "epochs": 10,
                        "batch_size": 8,
                        "learning_rate": 0.001,
                        "optimizer": "adam",
                    },
                    "data": {
                        "image_size": [512, 512],
                        "num_workers": 4,
                        "pin_memory": True,
                    },
                },
                "expected_outcome": "success",
                "performance_target": {
                    "setup_time": 2.0,
                    "validation_time": 1.0,
                },
            }
        )

        # RTX 3070 Ti optimized configuration
        scenarios.append(
            {
                "name": "rtx_3070_ti_optimized",
                "description": (
                    "Configuration optimized for RTX 3070 Ti VRAM constraints"
                ),
                "config": {
                    "model": {
                        "name": "unet",
                        "encoder": "efficientnet_b0",
                        "decoder": "unet",
                        "num_classes": 2,
                    },
                    "training": {
                        "epochs": 5,
                        "batch_size": 4,  # Reduced for 8GB VRAM
                        "learning_rate": 0.0005,
                        "optimizer": "adamw",
                        "mixed_precision": True,
                    },
                    "data": {
                        "image_size": [384, 384],  # Reduced size for memory
                        "num_workers": 2,
                        "pin_memory": True,
                    },
                },
                "expected_outcome": "success",
                "performance_target": {
                    "setup_time": 1.5,
                    "validation_time": 0.8,
                },
            }
        )

        # Invalid configuration scenarios
        scenarios.append(
            {
                "name": "invalid_missing_model",
                "description": "Configuration missing required model section",
                "config": {
                    "training": {
                        "epochs": 10,
                        "batch_size": 8,
                        "learning_rate": 0.001,
                    },
                    "data": {
                        "image_size": [512, 512],
                    },
                },
                "expected_outcome": "validation_error",
                "performance_target": {"error_detection_time": 0.5},
            }
        )

        scenarios.append(
            {
                "name": "invalid_batch_size",
                "description": "Configuration with invalid batch size",
                "config": {
                    "model": {
                        "name": "unet",
                        "encoder": "resnet50",
                        "num_classes": 2,
                    },
                    "training": {
                        "epochs": 10,
                        "batch_size": 0,  # Invalid batch size
                        "learning_rate": 0.001,
                    },
                    "data": {
                        "image_size": [512, 512],
                    },
                },
                "expected_outcome": "validation_error",
                "performance_target": {"error_detection_time": 0.3},
            }
        )

        return scenarios

    def generate_training_scenarios(self) -> list[dict[str, Any]]:
        """
        Generate training automation scenarios for testing. Returns: List of
        training scenarios with metadata
        """
        scenarios = []

        # Successful training scenario
        scenarios.append(
            {
                "name": "successful_training_minimal",
                "description": "Minimal training scenario that should succeed",
                "config": {
                    "model": {"name": "unet", "encoder": "resnet18"},
                    "training": {
                        "epochs": 2,
                        "batch_size": 2,
                        "learning_rate": 0.01,
                    },
                    "data": {"image_size": [256, 256]},
                },
                "expected_outcome": "success",
                "performance_target": {"training_time": 30.0},
            }
        )

        # VRAM exhaustion scenario
        scenarios.append(
            {
                "name": "vram_exhaustion",
                "description": (
                    "Scenario to trigger VRAM exhaustion on RTX 3070 Ti"
                ),
                "config": {
                    "model": {"name": "unet", "encoder": "swin_large"},
                    "training": {
                        "epochs": 1,
                        "batch_size": 32,  # Large batch size
                        "learning_rate": 0.001,
                    },
                    "data": {"image_size": [1024, 1024]},  # Large image size
                },
                "expected_outcome": "vram_error",
                "performance_target": {"error_detection_time": 10.0},
            }
        )

        # Training interruption scenario
        scenarios.append(
            {
                "name": "training_interruption",
                "description": "Training that gets interrupted mid-process",
                "config": {
                    "model": {"name": "unet", "encoder": "resnet50"},
                    "training": {
                        "epochs": 5,
                        "batch_size": 4,
                        "learning_rate": 0.001,
                        # Custom parameter for testing
                        "interrupt_after_epoch": 2,
                    },
                    "data": {"image_size": [512, 512]},
                },
                "expected_outcome": "interruption_handled",
                "performance_target": {"recovery_time": 5.0},
            }
        )

        return scenarios

    def generate_concurrent_scenarios(self) -> list[dict[str, Any]]:
        """
        Generate concurrent operation scenarios for testing. Returns: List of
        concurrent scenarios with metadata
        """
        scenarios = []

        # Multi-user simulation
        scenarios.append(
            {
                "name": "multi_user_config_loading",
                "description": (
                    "Multiple users loading configurations simultaneously"
                ),
                "user_count": 3,
                "operation": "config_loading",
                "duration_seconds": 30,
                "expected_outcome": "success",
                "performance_target": {"max_response_time": 3.0},
            }
        )

        # Resource contention
        scenarios.append(
            {
                "name": "resource_contention",
                "description": (
                    "Multiple processes competing for GPU resources"
                ),
                "process_count": 2,
                "operation": "training_setup",
                "duration_seconds": 60,
                "expected_outcome": "graceful_handling",
                "performance_target": {"resource_allocation_time": 10.0},
            }
        )

        return scenarios

    def save_scenario_to_file(
        self, scenario: dict[str, Any], filename: str
    ) -> Path:
        """
        Save a scenario configuration to a YAML file. Args: scenario: Scenario
        configuration dictionary filename: Name of the output file Returns:
        Path to the saved file
        """
        output_path = self.base_path / f"{filename}.yaml"

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(scenario, f, default_flow_style=False, sort_keys=False)

        return output_path

    def generate_test_artifacts(self) -> dict[str, list[Path]]:
        """
        Generate all test artifacts for automation scenarios. Returns:
        Dictionary mapping artifact types to file paths
        """
        artifacts: dict[str, list[Path]] = {
            "configuration_scenarios": [],
            "training_scenarios": [],
            "concurrent_scenarios": [],
        }

        # Generate configuration scenarios
        for scenario in self.generate_configuration_scenarios():
            path = self.save_scenario_to_file(
                scenario, f"config_{scenario['name']}"
            )
            artifacts["configuration_scenarios"].append(path)

        # Generate training scenarios
        for scenario in self.generate_training_scenarios():
            path = self.save_scenario_to_file(
                scenario, f"training_{scenario['name']}"
            )
            artifacts["training_scenarios"].append(path)

        # Generate concurrent scenarios
        for scenario in self.generate_concurrent_scenarios():
            path = self.save_scenario_to_file(
                scenario, f"concurrent_{scenario['name']}"
            )
            artifacts["concurrent_scenarios"].append(path)

        return artifacts

    def create_mock_training_data(self, scenario_name: str) -> dict[str, Path]:
        """
        Create mock training data for automation scenarios. Args:
        scenario_name: Name of the scenario for data generation Returns:
        Dictionary with paths to generated mock data
        """
        data_dir = self.base_path / "mock_data" / scenario_name
        data_dir.mkdir(parents=True, exist_ok=True)

        # Create unified mock data structure
        images_dir = data_dir / "images"
        images_dir.mkdir(exist_ok=True)

        masks_dir = data_dir / "masks"
        masks_dir.mkdir(exist_ok=True)

        # Create mock metadata files
        metadata = {
            "scenario": scenario_name,
            "generated_at": datetime.now().isoformat(),
            "data_type": "mock_automation_data",
            "image_count": 10,
            "image_size": [512, 512],
            "structure": "unified",
        }

        metadata_path = data_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return {
            "data_root": data_dir,
            "images_dir": images_dir,
            "masks_dir": masks_dir,
            "metadata": metadata_path,
        }

    def cleanup_test_artifacts(self, older_than_hours: int = 24) -> int:
        """
        Clean up old test artifacts to manage disk space. Args:
        older_than_hours: Remove artifacts older than this many hours Returns:
        Number of files removed
        """
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        removed_count = 0

        for file_path in self.base_path.rglob("*"):
            if file_path.is_file():
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < cutoff_time:
                    file_path.unlink()
                    removed_count += 1

        return removed_count
