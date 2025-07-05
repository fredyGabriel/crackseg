"""Integration test scenarios using modular workflow components.

This module contains the actual test scenarios that combine workflow
components to create comprehensive integration tests for user workflows
in the CrackSeg GUI.
"""

from .test_base import WorkflowTestBase
from .workflow_components import (
    ConfigurationWorkflowComponent,
    TrainingWorkflowComponent,
)


class TestWorkflowScenarios(WorkflowTestBase):
    """Complete integration test scenarios using modular workflow
    components."""

    def setup_method(self) -> None:
        """Setup workflow components for testing."""
        super().setup_method()
        self.config_workflow = ConfigurationWorkflowComponent(self)
        self.training_workflow = TrainingWorkflowComponent(self)

    def test_complete_configuration_workflow_success(self) -> None:
        """Test complete successful configuration workflow.

        Scenario: User loads valid configuration and sets up working directory.
        Expected: All workflow steps complete successfully.
        """
        # Create valid configuration
        config_file = self.create_valid_config_file()
        run_directory = self.temp_path / "run_outputs"

        # Execute configuration loading workflow
        config_result = self.config_workflow.execute_config_loading_workflow(
            config_file, expect_success=True
        )

        # Execute directory setup workflow
        dir_result = self.config_workflow.execute_directory_setup_workflow(
            run_directory, create_if_missing=True
        )

        # Verify successful completion
        assert config_result["success"]
        assert config_result["config_loaded"]
        assert config_result["session_state_updated"]
        assert dir_result["success"]
        assert dir_result["directory_created"]
        assert dir_result["subdirectories_created"]

    def test_configuration_workflow_error_handling(self) -> None:
        """Test configuration workflow with invalid inputs.

        Scenario: User attempts to load invalid configuration file.
        Expected: Proper error handling and informative error messages.
        """
        # Create invalid configuration
        invalid_config = self.create_invalid_config_file()

        # Execute configuration loading workflow (expect failure)
        config_result = self.config_workflow.execute_config_loading_workflow(
            invalid_config, expect_success=False
        )

        # Verify proper error handling
        assert not config_result["success"]
        assert not config_result["config_loaded"]
        assert len(config_result["validation_errors"]) > 0
        assert "YAML parsing error" in str(config_result["validation_errors"])

    def test_complete_training_preparation_workflow(self) -> None:
        """Test complete workflow from configuration to training readiness.

        Scenario: User prepares complete environment for training from scratch.
        Expected: All preparation steps complete successfully.
        """
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
        assert training_result["config_valid"]
        assert training_result["environment_ready"]
        assert training_result["logging_configured"]

    def test_end_to_end_training_simulation_workflow(self) -> None:
        """Test complete end-to-end workflow including simulated training.

        Scenario: User executes complete training workflow from start to
        finish.
        Expected: All workflow phases complete with realistic outcomes.
        """
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

        # Execute monitoring workflow
        monitoring_result = (
            self.training_workflow.execute_training_monitoring_workflow(
                run_directory
            )
        )

        # Verify complete end-to-end workflow
        assert all(
            [
                config_result["success"],
                dir_result["success"],
                setup_result["success"],
                training_result["success"],
                monitoring_result["success"],
            ]
        )

        # Verify training metrics generated
        assert "final_loss" in training_result["final_metrics"]
        assert "final_iou" in training_result["final_metrics"]
        assert training_result["final_metrics"]["epochs_completed"] == 5

        # Verify monitoring capabilities
        assert monitoring_result["logs_available"]
        assert monitoring_result["metrics_tracked"]
        assert monitoring_result["tensorboard_ready"]

    def test_workflow_error_scenarios(self) -> None:
        """Test various error scenarios in workflows.

        Scenario: Multiple error conditions that users might encounter.
        Expected: Graceful error handling for all error scenarios.
        """
        # Test missing configuration file
        missing_file = self.temp_path / "nonexistent.yaml"
        config_result = self.config_workflow.execute_config_loading_workflow(
            missing_file, expect_success=False
        )
        assert not config_result["success"]
        assert "File does not exist" in config_result["validation_errors"]

        # Test training setup without proper configuration
        incomplete_training_config = {
            "model": {"name": "unet"}
        }  # Missing training section
        nonexistent_dir = self.temp_path / "missing_directory"
        training_result = (
            self.training_workflow.execute_training_setup_workflow(
                incomplete_training_config, nonexistent_dir
            )
        )
        assert not training_result["success"]
        assert not training_result["environment_ready"]

    def test_workflow_component_modularity(self) -> None:
        """Test that workflow components can be used independently and
        combined.

        Scenario: Components used in isolation and different combinations.
        Expected: Flexible usage patterns with consistent behavior.
        """
        # Test independent usage of components
        config_file = self.create_valid_config_file()

        # Use configuration component independently
        config_only_result = (
            self.config_workflow.execute_config_loading_workflow(config_file)
        )
        assert config_only_result["success"]

        # Combine components in different orders
        run_dir = self.temp_path / "modular_test"
        dir_result = self.config_workflow.execute_directory_setup_workflow(
            run_dir, True
        )

        # Verify independent and combined usage works
        assert config_only_result["success"]
        assert dir_result["success"]


class TestWorkflowPerformance(WorkflowTestBase):
    """Performance testing for workflow scenarios."""

    def setup_method(self) -> None:
        """Setup performance testing environment."""
        super().setup_method()
        self.config_workflow = ConfigurationWorkflowComponent(self)

    def test_workflow_performance_characteristics(self) -> None:
        """Test that workflows complete within reasonable time limits.

        Scenario: Performance validation for configuration workflows.
        Expected: Workflows complete within acceptable time bounds.
        """
        import time

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
        assert (
            execution_time < 1.0
        ), f"Workflow took {execution_time:.2f}s, expected < 1.0s"
