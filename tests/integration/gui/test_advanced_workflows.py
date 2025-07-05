"""Advanced end-to-end workflow integration tests for CrackSeg GUI.

This module contains complex integration test scenarios that combine multiple
workflow components to test complete user journeys.
"""

from .test_base import WorkflowTestBase
from .workflow_components import (
    ConfigurationWorkflowComponent,
    TrainingWorkflowComponent,
)


class TestAdvancedWorkflowScenarios(WorkflowTestBase):
    """Advanced integration test scenarios for complete user workflows."""

    def setup_method(self) -> None:
        """Setup workflow components for testing."""
        super().setup_method()
        self.config_workflow = ConfigurationWorkflowComponent(self)
        self.training_workflow = TrainingWorkflowComponent(self)

    def test_complete_training_preparation_workflow(self) -> None:
        """Test complete workflow from configuration to training readiness."""
        # Step 1: Setup configuration
        config_content = {
            "model": {"name": "unet", "encoder": "resnet50"},
            "training": {
                "epochs": 20,
                "learning_rate": 0.001,
                "batch_size": 8,
            },
            "data": {"image_size": [512, 512]},
        }
        config_file = self.create_valid_config_file(config_content)

        # Step 2: Load configuration
        config_result = self.config_workflow.execute_config_loading_workflow(
            config_file, expect_success=True
        )

        # Step 3: Setup working directory
        run_directory = self.temp_path / "training_run"
        dir_result = self.config_workflow.execute_directory_setup_workflow(
            run_directory, create_if_missing=True
        )

        # Step 4: Setup training environment
        training_result = (
            self.training_workflow.execute_training_setup_workflow(
                config_content, run_directory
            )
        )

        # Verify complete workflow success
        assert config_result["success"]
        assert dir_result["success"]
        assert training_result["success"]
        assert training_result["ready_for_training"]

    def test_end_to_end_training_simulation_workflow(self) -> None:
        """Test complete end-to-end workflow including simulated training."""
        # Complete setup workflow
        config_content = {
            "model": {"name": "unet", "encoder": "resnet50"},
            "training": {"epochs": 5, "learning_rate": 0.001},
        }
        config_file = self.create_valid_config_file(config_content)
        run_directory = self.temp_path / "e2e_training"

        # Execute complete workflow
        config_result = self.config_workflow.execute_config_loading_workflow(
            config_file
        )
        dir_result = self.config_workflow.execute_directory_setup_workflow(
            run_directory, True
        )
        setup_result = self.training_workflow.execute_training_setup_workflow(
            config_content, run_directory
        )

        # Execute simulated training
        training_result = (
            self.training_workflow.simulate_training_execution_workflow(
                config_content["training"]
            )
        )

        # Verify complete end-to-end workflow
        assert all(
            [
                config_result["success"],
                dir_result["success"],
                setup_result["success"],
                training_result["success"],
            ]
        )

        # Verify training metrics generated
        assert "final_loss" in training_result["final_metrics"]
        assert "final_iou" in training_result["final_metrics"]
        assert training_result["final_metrics"]["epochs_completed"] == 5

    def test_complex_configuration_workflow(self) -> None:
        """Test workflow with complex configuration and multiple components."""
        # Create complex configuration
        complex_config = {
            "model": {
                "name": "swin_unet",
                "encoder": {
                    "type": "swin_transformer",
                    "pretrained": True,
                    "depths": [2, 2, 6, 2],
                },
            },
            "training": {
                "epochs": 50,
                "learning_rate": 0.0001,
                "batch_size": 4,
                "optimizer": "adamw",
                "loss_functions": [
                    {"name": "dice_loss", "weight": 0.5},
                    {"name": "focal_loss", "weight": 0.3},
                ],
            },
            "data": {
                "image_size": [512, 512],
                "augmentations": {"horizontal_flip": True, "rotation": True},
            },
        }

        config_file = self.create_valid_config_file(complex_config)
        run_directory = self.temp_path / "complex_training"

        # Execute complete complex workflow
        config_result = self.config_workflow.execute_config_loading_workflow(
            config_file
        )
        dir_result = self.config_workflow.execute_directory_setup_workflow(
            run_directory, True
        )
        setup_result = self.training_workflow.execute_training_setup_workflow(
            complex_config, run_directory
        )

        # Verify all steps successful
        assert config_result["success"]
        assert dir_result["success"]
        assert setup_result["success"]

    def test_workflow_with_monitoring_components(self) -> None:
        """Test workflow that includes training monitoring components."""
        # Setup basic training workflow
        config_content = {
            "model": {"name": "unet"},
            "training": {"epochs": 10, "learning_rate": 0.001},
        }
        run_directory = self.temp_path / "monitored_training"
        run_directory.mkdir(parents=True, exist_ok=True)

        # Setup training
        setup_result = self.training_workflow.execute_training_setup_workflow(
            config_content, run_directory
        )

        # Execute monitoring workflow
        monitoring_result = (
            self.training_workflow.execute_training_monitoring_workflow(
                run_directory, expected_metrics=["loss", "iou", "dice"]
            )
        )

        # Verify monitoring setup
        assert setup_result["success"]
        assert monitoring_result["success"]
        assert monitoring_result["logs_available"]
        assert monitoring_result["tensorboard_ready"]

    def test_multi_configuration_workflow(self) -> None:
        """Test workflow with multiple configurations for comparison."""
        configs = [
            {
                "model": {"name": "unet", "encoder": "resnet50"},
                "training": {"epochs": 10, "learning_rate": 0.001},
            },
            {
                "model": {"name": "deeplabv3", "encoder": "resnet101"},
                "training": {"epochs": 10, "learning_rate": 0.0005},
            },
        ]

        results = []
        for i, config in enumerate(configs):
            config_file = self.create_valid_config_file(config)
            run_dir = self.temp_path / f"multi_config_{i}"

            # Execute workflow for each config
            config_result = (
                self.config_workflow.execute_config_loading_workflow(
                    config_file
                )
            )
            dir_result = self.config_workflow.execute_directory_setup_workflow(
                run_dir, True
            )
            setup_result = (
                self.training_workflow.execute_training_setup_workflow(
                    config, run_dir
                )
            )

            results.append(
                {
                    "config": config_result["success"],
                    "directory": dir_result["success"],
                    "setup": setup_result["success"],
                }
            )

        # Verify all configurations processed successfully
        assert all(all(result.values()) for result in results)
        assert len(results) == 2
