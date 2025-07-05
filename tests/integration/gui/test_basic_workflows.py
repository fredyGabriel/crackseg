"""Basic workflow integration tests for CrackSeg GUI.

This module contains basic integration test scenarios that test individual
workflow components and simple combinations.
"""

from .test_base import WorkflowTestBase
from .workflow_components import (
    ConfigurationWorkflowComponent,
    TrainingWorkflowComponent,
)


class TestBasicWorkflowScenarios(WorkflowTestBase):
    """Basic integration test scenarios using modular workflow components."""

    def setup_method(self) -> None:
        """Setup workflow components for testing."""
        super().setup_method()
        self.config_workflow = ConfigurationWorkflowComponent(self)
        self.training_workflow = TrainingWorkflowComponent(self)

    def test_complete_configuration_workflow_success(self) -> None:
        """Test complete successful configuration workflow."""
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
        assert dir_result["success"]
        assert dir_result["directory_created"]
        assert dir_result["subdirectories_created"]

    def test_configuration_workflow_invalid_file(self) -> None:
        """Test configuration workflow with invalid file."""
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

    def test_basic_training_setup_workflow(self) -> None:
        """Test basic training setup workflow."""
        # Setup configuration
        config_content = {
            "model": {"name": "unet", "encoder": "resnet50"},
            "training": {
                "epochs": 5,
                "learning_rate": 0.001,
                "batch_size": 4,
            },
            "data": {"image_size": [512, 512]},
        }
        run_directory = self.temp_path / "training_run"
        run_directory.mkdir(parents=True, exist_ok=True)

        # Execute training setup workflow
        training_result = (
            self.training_workflow.execute_training_setup_workflow(
                config_content, run_directory
            )
        )

        # Verify successful setup
        assert training_result["success"]
        assert training_result["config_valid"]
        assert training_result["environment_ready"]
        assert training_result["logging_configured"]
        assert training_result["ready_for_training"]

    def test_workflow_error_scenarios(self) -> None:
        """Test various error scenarios in workflows."""
        # Test missing configuration file
        missing_file = self.temp_path / "nonexistent.yaml"
        config_result = self.config_workflow.execute_config_loading_workflow(
            missing_file, expect_success=False
        )
        assert not config_result["success"]
        assert "File does not exist" in config_result["validation_errors"]

        # Test training setup without proper configuration
        incomplete_training_config = {
            "model": {"name": "unet"}  # Missing training section
        }
        nonexistent_dir = self.temp_path / "missing_directory"
        training_result = (
            self.training_workflow.execute_training_setup_workflow(
                incomplete_training_config, nonexistent_dir
            )
        )
        assert not training_result["success"]
        assert not training_result["environment_ready"]

    def test_workflow_component_modularity(self) -> None:
        """Test that workflow components can be used independently."""
        # Test independent usage of components
        config_file = self.create_valid_config_file()

        # Use configuration component independently
        config_only_result = (
            self.config_workflow.execute_config_loading_workflow(config_file)
        )
        assert config_only_result["success"]

        # Use directory setup independently
        run_dir = self.temp_path / "modular_test"
        dir_result = self.config_workflow.execute_directory_setup_workflow(
            run_dir, True
        )

        # Verify independent usage works
        assert config_only_result["success"]
        assert dir_result["success"]
