"""Main concurrent operation tests.

Primary entry point for concurrent operation testing. Contains basic
concurrent operation tests. Complex scenarios moved to specialized
test files in concurrent_tests/ module.
"""

from typing import Any

import yaml

from .test_base import WorkflowTestBase
from .workflow_components.concurrent import ConcurrentOperationMixin
from .workflow_components.config_workflow import ConfigurationWorkflowComponent
from .workflow_components.error_scenario_mixin import ErrorScenarioMixin
from .workflow_components.session_state_mixin import SessionStateMixin
from .workflow_components.training_workflow import TrainingWorkflowComponent


class TestConcurrentOperations(
    WorkflowTestBase,
    ConcurrentOperationMixin,
    ErrorScenarioMixin,
    SessionStateMixin,
):
    """Main test class for concurrent operations."""

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

        # Create basic test configurations
        self._create_basic_test_configs()

    def teardown_method(self) -> None:
        """Clean up after test method."""
        self.cleanup_concurrent_operations()
        super().teardown_method()

    def test_basic_concurrent_operations(self) -> None:
        """Test basic concurrent operation functionality."""

        def basic_workflow(user_id: int) -> dict[str, Any]:
            """Basic workflow for concurrent testing."""
            return {
                "user_id": user_id,
                "operation": "basic_test",
                "success": True,
                "timestamp": user_id * 0.01,
            }

        # Test basic multi-user workflow
        result = self.simulate_multi_user_workflow(
            user_count=3,
            workflow_function=basic_workflow,
        )

        # Verify basic concurrent operations work
        assert result["success"], (
            "Basic concurrent operations failed: "
            f"{result['concurrent_issues']}"
        )
        assert result["workflows_completed"] == 3
        assert result["workflows_failed"] == 0

    def test_concurrent_config_access(self) -> None:
        """Test concurrent access to configuration files."""

        def config_access_workflow(user_id: int) -> dict[str, Any]:
            """Workflow for concurrent config access."""
            config_path = self.temp_path / f"basic_config_{user_id % 2}.yaml"

            result = self.config_component.execute_config_loading_workflow(
                config_path
            )
            result["user_id"] = user_id
            return result

        # Test concurrent config access
        result = self.simulate_multi_user_workflow(
            user_count=4,
            workflow_function=config_access_workflow,
        )

        # Verify concurrent config access
        assert result[
            "success"
        ], f"Concurrent config access failed: {result['concurrent_issues']}"
        assert result["workflows_completed"] == 4

    def test_resource_coordination(self) -> None:
        """Test basic resource coordination between concurrent operations."""
        # Simple resource contention test
        simple_scenarios = [
            {"duration": 0.1},
            {"duration": 0.2},
            {"duration": 0.1},
        ]

        result = self.test_resource_contention(
            resource_name="test_resource",
            contention_scenarios=simple_scenarios,
            max_concurrent_access=2,
        )

        # Verify resource coordination
        assert result[
            "success"
        ], f"Resource coordination failed: {result['contention_violations']}"
        assert result["scenarios_passed"] >= 2

    def test_basic_process_sync(self) -> None:
        """Test basic process synchronization."""
        result = self.test_process_synchronization(
            sync_scenario="basic_sync",
            process_count=3,
            sync_points=["start", "middle", "end"],
        )

        # Verify basic synchronization
        assert result[
            "success"
        ], f"Basic process sync failed: {result['sync_violations']}"
        assert result["processes_completed"] == 3

    def test_basic_system_stability(self) -> None:
        """Test basic system stability under light load."""
        light_scenarios = [
            {"type": "cpu_intensive", "duration": 0.1},
            {"type": "memory_intensive", "duration": 0.1},
        ]

        result = self.test_system_stability_under_load(
            load_scenarios=light_scenarios,
            monitoring_duration=1.0,
        )

        # Verify basic stability
        assert result[
            "success"
        ], f"Basic stability failed: {result['stability_violations']}"

        metrics = result["system_metrics"]
        assert metrics["error_count"] == 0

    def test_basic_data_integrity(self) -> None:
        """Test basic data integrity during concurrent operations."""
        simple_operations = [
            {"type": "write", "id": "op1"},
            {"type": "read", "id": "op2"},
            {"type": "write", "id": "op3"},
        ]

        result = self.test_data_integrity_under_concurrency(
            data_operations=simple_operations
        )

        # Verify basic data integrity
        assert result[
            "success"
        ], f"Basic data integrity failed: {result['integrity_violations']}"
        assert result["operations_completed"] == 3
        assert result["operations_failed"] == 0

    def _create_basic_test_configs(self) -> None:
        """Create basic test configuration files."""
        basic_config = {
            "model": {
                "name": "basic_test_model",
                "input_size": 256,
                "num_classes": 2,
            },
            "training": {
                "epochs": 3,
                "batch_size": 8,
                "learning_rate": 0.01,
            },
        }

        # Create basic test configs
        for i in range(3):
            config_path = self.temp_path / f"basic_config_{i}.yaml"
            config: dict[str, Any] = basic_config.copy()
            config["model"]["name"] = f"basic_test_model_{i}"

            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f)
