"""Multi-user concurrent operation tests.

Tests focused on multi-user workflows, error recovery, and performance
under concurrent load. Extracted from oversized test_concurrent_operations.py.
"""

import time
from pathlib import Path
from typing import Any

import yaml

from ..test_base import WorkflowTestBase
from ..workflow_components.concurrent import ConcurrentOperationMixin
from ..workflow_components.config_workflow import (
    ConfigurationWorkflowComponent,
)
from ..workflow_components.error_scenario_mixin import ErrorScenarioMixin
from ..workflow_components.session_state_mixin import SessionStateMixin
from ..workflow_components.training_workflow import TrainingWorkflowComponent


class TestMultiUserOperations(
    WorkflowTestBase,
    ConcurrentOperationMixin,
    ErrorScenarioMixin,
    SessionStateMixin,
):
    """Test concurrent operations with multiple users."""

    def setup_method(self) -> None:
        """Set up test method with concurrent operation components."""
        super().setup_method()

        # Initialize workflow components
        self.config_component = ConfigurationWorkflowComponent(
            test_utilities=self
        )
        self.training_component = TrainingWorkflowComponent(
            test_utilities=self
        )

        # Create test configurations
        self._create_test_configs()

    def teardown_method(self) -> None:
        """Clean up after test method."""
        self.cleanup_concurrent_operations()
        super().teardown_method()

    def test_multi_user_config_loading(self) -> None:
        """Test multiple users loading configs simultaneously."""

        def user_config_workflow(user_id: int) -> dict[str, Any]:
            """Workflow for individual user config loading."""
            config_path = self.temp_path / f"test_config_{user_id}.yaml"

            result = self.config_component.execute_config_loading_workflow(
                config_path
            )
            result["user_id"] = user_id
            return result

        # Test with 5 concurrent users
        result = self.simulate_multi_user_workflow(
            user_count=5,
            workflow_function=user_config_workflow,
        )

        # Verify multi-user workflow success
        assert result[
            "success"
        ], f"Multi-user config loading failed: {result['concurrent_issues']}"
        assert result["workflows_completed"] == 5
        assert result["workflows_failed"] == 0

        # Verify all users completed successfully
        for user_result in result["user_results"].values():
            assert isinstance(user_result, dict)
            assert user_result[
                "success"
            ], f"User workflow failed: {user_result}"

        # Verify reasonable timing
        if result["execution_times"]:
            max_time = result["max_execution_time"]
            assert (
                max_time < 2.0
            ), f"Config loading took too long: {max_time:.3f}s"

    def test_concurrent_training_processes(self) -> None:
        """Test concurrent training process setup."""

        def user_training_workflow(user_id: int) -> dict[str, Any]:
            """Workflow for individual user training setup."""
            run_dir = self.temp_path / f"training_user_{user_id}"
            run_dir.mkdir(parents=True, exist_ok=True)

            # Create config for this user
            config = {
                "model": {"name": f"model_{user_id}"},
                "training": {"epochs": 3, "batch_size": 4},
            }

            return {
                "user_id": user_id,
                "config": config,
                "run_dir": str(run_dir),
                "success": True,
            }

        # Test with 3 concurrent training users
        result = self.simulate_multi_user_workflow(
            user_count=3,
            workflow_function=user_training_workflow,
        )

        # Verify concurrent training setup
        assert result[
            "success"
        ], f"Concurrent training failed: {result['concurrent_issues']}"
        assert result["workflows_completed"] == 3
        assert result["workflows_failed"] == 0

        # Verify training directories were created
        for user_result in result["user_results"].values():
            assert isinstance(user_result, dict)
            run_dir = Path(user_result["run_dir"])
            assert (
                run_dir.exists()
            ), f"Training directory not created: {run_dir}"

    def test_concurrent_error_recovery(self) -> None:
        """Test error recovery mechanisms under concurrent conditions."""

        def error_prone_workflow(user_id: int) -> dict[str, Any]:
            """Workflow that may encounter errors."""
            if user_id == 2:  # Introduce error for specific user
                corrupted_config = self.create_corrupted_config_file(
                    "yaml_syntax"
                )
                result = self.config_component.execute_config_loading_workflow(
                    corrupted_config, expect_success=False
                )
            else:
                config_path = self.temp_path / f"valid_config_{user_id}.yaml"
                config_content = {
                    "model": {"name": f"model_{user_id}"},
                    "training": {"epochs": 5},
                }
                with open(config_path, "w", encoding="utf-8") as f:
                    yaml.dump(config_content, f)

                result = self.config_component.execute_config_loading_workflow(
                    config_path
                )

            result["user_id"] = user_id
            return result

        # Test error recovery with mixed success/failure scenarios
        result = self.simulate_multi_user_workflow(
            user_count=4,
            workflow_function=error_prone_workflow,
        )

        # Verify error isolation - one user's error doesn't affect others
        assert result["workflows_completed"] >= 3, "Error affected other users"
        assert result["workflows_failed"] <= 1, "Too many failures"

        # Verify successful users completed normally
        successful_users = [
            user_result
            for user_result in result["user_results"].values()
            if isinstance(user_result, dict)
            and user_result.get("success", False)
        ]
        assert len(successful_users) >= 3, "Not enough successful users"

    def test_performance_under_concurrent_load(self) -> None:
        """Test performance characteristics under concurrent load."""

        def performance_workflow(user_id: int) -> dict[str, Any]:
            """Workflow to measure performance under load."""
            start_time = time.time()

            # Simulate realistic workflow steps
            config_path = self.temp_path / f"perf_config_{user_id}.yaml"
            config_content = {
                "model": {
                    "name": f"perf_model_{user_id}",
                    "layers": user_id + 5,
                },
                "training": {"epochs": 3, "batch_size": 8},
            }

            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config_content, f)

            config_result = (
                self.config_component.execute_config_loading_workflow(
                    config_path
                )
            )

            # Simulate directory setup
            run_dir = self.temp_path / f"perf_run_{user_id}"
            dir_result = (
                self.config_component.execute_directory_setup_workflow(run_dir)
            )

            total_time = time.time() - start_time

            return {
                "user_id": user_id,
                "config_success": config_result["success"],
                "dir_success": dir_result["success"],
                "total_time": total_time,
                "success": config_result["success"] and dir_result["success"],
            }

        # Test performance with 6 concurrent users
        result = self.simulate_multi_user_workflow(
            user_count=6,
            workflow_function=performance_workflow,
        )

        # Verify performance standards
        assert result[
            "success"
        ], f"Performance test failed: {result['concurrent_issues']}"
        assert result["workflows_completed"] == 6

        # Performance thresholds
        if result["execution_times"]:
            max_time = result["max_execution_time"]
            avg_time = result["avg_execution_time"]

            assert (
                max_time < 3.0
            ), f"Maximum execution time too high: {max_time:.3f}s"
            assert (
                avg_time < 1.0
            ), f"Average execution time too high: {avg_time:.3f}s"

    def _create_test_configs(self) -> None:
        """Create test configuration files for concurrent testing."""
        base_config: dict[str, Any] = {
            "model": {
                "name": "test_model",
                "input_size": 512,
                "num_classes": 2,
            },
            "training": {
                "epochs": 5,
                "batch_size": 16,
                "learning_rate": 0.001,
            },
        }

        # Create multiple test configs
        for i in range(10):
            config_path = self.temp_path / f"test_config_{i}.yaml"
            config = base_config.copy()
            config["model"]["name"] = f"test_model_{i}"
            config["training"]["batch_size"] = 8 + (i * 2)

            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f)
