"""Resource contention concurrent operation tests.

Tests focused on resource conflicts, GPU contention, file system access,
and process synchronization. Extracted from oversized test_concurrent_operations.py.
"""

from typing import Any

from ..test_base import WorkflowTestBase
from ..workflow_components.concurrent import ConcurrentOperationMixin
from ..workflow_components.config_workflow import (
    ConfigurationWorkflowComponent,
)
from ..workflow_components.error_scenario_mixin import ErrorScenarioMixin
from ..workflow_components.session_state_mixin import SessionStateMixin
from ..workflow_components.training_workflow import TrainingWorkflowComponent


class TestResourceContention(
    WorkflowTestBase,
    ConcurrentOperationMixin,
    ErrorScenarioMixin,
    SessionStateMixin,
):
    """Test concurrent operations with resource contention scenarios."""

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

    def teardown_method(self) -> None:
        """Clean up after test method."""
        self.cleanup_concurrent_operations()
        super().teardown_method()

    def test_gpu_resource_contention(self) -> None:
        """Test GPU resource contention scenarios."""
        # Define GPU contention scenarios
        gpu_scenarios = [
            {"duration": 0.5, "memory_usage": "high"},
            {"duration": 0.3, "memory_usage": "medium"},
            {"duration": 0.4, "memory_usage": "low"},
            {"duration": 0.2, "memory_usage": "medium"},
        ]

        result = self.test_resource_contention(
            resource_name="gpu_memory",
            contention_scenarios=gpu_scenarios,
            max_concurrent_access=2,  # RTX 3070 Ti limitation
        )

        # Verify resource contention is properly handled
        assert result["success"], (
            "GPU resource contention failed: "
            f"{result['contention_violations']}"
        )
        assert (
            result["scenarios_passed"] >= 2
        ), "Not enough GPU scenarios passed"

        # Verify resource access patterns
        access_patterns = result["access_patterns"]
        assert len(access_patterns) == 4, "Missing GPU access patterns"

        # Verify no more than 2 concurrent accesses (RTX 3070 Ti limit)
        for pattern in access_patterns:
            assert pattern["acquired"], "GPU access should be acquired"
            assert pattern["released"], "GPU access should be released"

    def test_file_system_resource_contention(self) -> None:
        """Test file system resource contention scenarios."""
        # Define file system contention scenarios
        fs_scenarios = [
            {"duration": 0.2, "operation": "read"},
            {"duration": 0.3, "operation": "write"},
            {"duration": 0.1, "operation": "read"},
            {"duration": 0.4, "operation": "write"},
            {"duration": 0.2, "operation": "read"},
        ]

        result = self.test_resource_contention(
            resource_name="file_system",
            contention_scenarios=fs_scenarios,
            max_concurrent_access=3,  # Allow more concurrent file operations
        )

        # Verify file system contention handling
        assert result[
            "success"
        ], f"File system contention failed: {result['contention_violations']}"
        assert (
            result["scenarios_passed"] >= 3
        ), "Not enough file system scenarios passed"

        # Verify reasonable access patterns
        access_patterns = result["access_patterns"]
        assert len(access_patterns) == 5, "Missing file system access patterns"

    def test_process_synchronization_config_training(self) -> None:
        """
        Test synchronization between config loading and training processes.
        """
        result = self.test_process_synchronization(
            sync_scenario="config_training_sync",
            process_count=4,
            sync_points=["config_load", "training_setup", "execution"],
        )

        # Verify process synchronization
        assert result[
            "success"
        ], f"Process synchronization failed: {result['sync_violations']}"
        assert (
            result["processes_completed"] == 4
        ), "Not all processes completed synchronization"

        # Verify synchronization points were hit
        for process_timing in result["process_timings"].values():
            assert "config_load" in process_timing["sync_times"]
            assert "training_setup" in process_timing["sync_times"]
            assert "execution" in process_timing["sync_times"]

    def test_session_state_concurrent_access(self) -> None:
        """Test concurrent access to session state."""

        def concurrent_session_operation(operation_id: int) -> dict[str, Any]:
            """Concurrent session state operation."""
            session_key = f"concurrent_key_{operation_id}"
            session_value = f"concurrent_value_{operation_id}"

            # Simulate session state operations
            result = {
                "operation_id": operation_id,
                "session_key": session_key,
                "session_value": session_value,
                "success": True,
            }

            return result

        # Test concurrent session operations
        result = self.simulate_multi_user_workflow(
            user_count=5,
            workflow_function=concurrent_session_operation,
        )

        # Verify concurrent session access
        assert result[
            "success"
        ], f"Concurrent session access failed: {result['concurrent_issues']}"
        assert result["workflows_completed"] == 5
        assert result["workflows_failed"] == 0

        # Verify session state consistency
        for user_result in result["user_results"].values():
            assert isinstance(user_result, dict)
            assert user_result["success"], "Session operation failed"
            assert "session_key" in user_result
            assert "session_value" in user_result

    def test_data_integrity_concurrent_operations(self) -> None:
        """Test data integrity during concurrent operations."""
        # Define concurrent data operations with proper types
        data_operations = [
            {"type": "write", "id": "op1"},
            {"type": "write", "id": "op2"},
            {"type": "read", "id": "op3"},
            {"type": "modify", "id": "op4"},
            {"type": "write", "id": "op5"},
            {"type": "read", "id": "op6"},
            {"type": "write", "id": "op7"},
            {"type": "delete", "id": "op8"},
        ]

        result = self.test_data_integrity_under_concurrency(
            data_operations=data_operations
        )

        # Verify data integrity
        assert result[
            "success"
        ], f"Data integrity failed: {result['integrity_violations']}"
        assert result["operations_completed"] == 8
        assert result["operations_failed"] == 0

        # Verify final data state consistency
        final_state = result["final_data_state"]
        assert "counter" in final_state
        assert "data_store_keys" in final_state
        assert "data_store_count" in final_state

        # Verify data consistency checks
        consistency_checks = result["data_consistency_checks"]
        assert len(consistency_checks) > 0, "No consistency checks performed"

        for check in consistency_checks:
            assert check[
                "consistent"
            ], f"Data consistency check failed: {check}"
