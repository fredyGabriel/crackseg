"""Performance testing for GUI workflow scenarios.

This module contains performance tests for workflow scenarios to ensure
they complete within acceptable time limits.
"""

import time

from .test_base import WorkflowTestBase
from .workflow_components import ConfigurationWorkflowComponent


class TestWorkflowPerformance(WorkflowTestBase):
    """Performance testing for workflow scenarios."""

    def setup_method(self) -> None:
        """Setup performance testing environment."""
        super().setup_method()
        self.config_workflow = ConfigurationWorkflowComponent(self)

    def test_workflow_performance_characteristics(self) -> None:
        """Test that workflows complete within reasonable time limits."""
        # Create test configuration
        config_file = self.create_valid_config_file()

        # Measure configuration loading workflow performance
        start_time = time.time()
        config_result = self.config_workflow.execute_config_loading_workflow(
            config_file
        )
        end_time = time.time()

        # Verify success and performance
        assert config_result["success"]
        execution_time = end_time - start_time

        # Configuration workflow should complete quickly (< 1 second for test
        # files)
        assert execution_time < 1.0, (
            f"Configuration workflow took {execution_time:.2f}s, expected < "
            "1.0s"
        )

    def test_large_configuration_workflow_performance(self) -> None:
        """Test workflow performance with larger configuration files."""
        # Create a larger configuration file
        large_config = {
            "model": {
                "name": "swin_unet",
                "encoder": {
                    "type": "swin_transformer",
                    "pretrained": True,
                    "window_size": 7,
                    "patch_size": 4,
                    "depths": [2, 2, 6, 2],
                    "num_heads": [3, 6, 12, 24],
                },
            },
            "training": {
                "epochs": 100,
                "learning_rate": 0.0001,
                "batch_size": 4,
                "optimizer": "adamw",
                "scheduler": "cosine",
                "loss_functions": [
                    {"name": "dice_loss", "weight": 0.5},
                    {"name": "focal_loss", "weight": 0.3},
                    {"name": "boundary_loss", "weight": 0.2},
                ],
            },
            "data": {
                "image_size": [512, 512],
                "augmentations": {
                    "horizontal_flip": True,
                    "vertical_flip": True,
                    "rotation": {"limit": 15},
                    "brightness": {"limit": 0.2},
                    "contrast": {"limit": 0.2},
                },
            },
        }

        large_config_file = self.create_valid_config_file(large_config)

        # Test that even large configurations are processed efficiently
        start_time = time.time()
        config_result = self.config_workflow.execute_config_loading_workflow(
            large_config_file
        )
        end_time = time.time()

        # Verify success and reasonable performance
        assert config_result["success"]
        execution_time = end_time - start_time
        assert execution_time < 2.0, (
            f"Large config workflow took {execution_time:.2f}s, expected < "
            "2.0s"
        )

    def test_directory_setup_performance(self) -> None:
        """Test performance of directory setup workflows."""
        run_directory = self.temp_path / "performance_test"

        # Measure directory setup performance
        start_time = time.time()
        dir_result = self.config_workflow.execute_directory_setup_workflow(
            run_directory, create_if_missing=True
        )
        end_time = time.time()

        # Verify success and performance
        assert dir_result["success"]
        execution_time = end_time - start_time

        # Directory setup should be very fast (< 0.5 seconds)
        assert (
            execution_time < 0.5
        ), f"Directory setup took {execution_time:.2f}s, expected < 0.5s"

    def test_workflow_scalability(self) -> None:
        """Test workflow performance scalability with multiple operations."""
        # Test multiple configuration loading operations
        num_configs = 10
        config_files = []

        # Create multiple config files
        setup_start = time.time()
        for i in range(num_configs):
            config_content = {
                "model": {"name": f"model_{i}", "encoder": "resnet50"},
                "training": {"epochs": 10 + i, "learning_rate": 0.001},
            }
            config_file = self.create_valid_config_file(config_content)
            config_files.append(config_file)
        setup_end = time.time()

        # Process all configurations
        process_start = time.time()
        results = []
        for config_file in config_files:
            result = self.config_workflow.execute_config_loading_workflow(
                config_file
            )
            results.append(result)
        process_end = time.time()

        # Verify all successful
        assert all(result["success"] for result in results)
        assert len(results) == num_configs

        # Check scalability metrics
        setup_time = setup_end - setup_start
        process_time = process_end - process_start
        avg_time_per_config = process_time / num_configs

        # Performance should scale reasonably
        assert (
            setup_time < 2.0
        ), f"Setup took {setup_time:.2f}s, expected < 2.0s"
        assert (
            process_time < 5.0
        ), f"Processing took {process_time:.2f}s, expected < 5.0s"
        assert avg_time_per_config < 0.5, (
            f"Average time per config {avg_time_per_config:.2f}s, expected < "
            "0.5s"
        )
