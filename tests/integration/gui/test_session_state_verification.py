"""Session state verification tests for integration testing.

This module implements comprehensive session state testing that builds upon
the workflow and error testing frameworks from tasks 9.1 and 9.2.
"""

from typing import Any

from tests.integration.gui.test_base import WorkflowTestBase
from tests.integration.gui.workflow_components.config_workflow import (
    ConfigurationWorkflowComponent,
)
from tests.integration.gui.workflow_components.session_state_mixin import (
    SessionStateMixin,
)
from tests.integration.gui.workflow_components.training_workflow import (
    TrainingWorkflowComponent,
)


class TestSessionStateVerification(WorkflowTestBase, SessionStateMixin):
    """Comprehensive session state verification tests.

    Extends the base integration test class and includes session state
    testing capabilities through the SessionStateMixin.
    """

    def setup_method(self) -> None:
        """Set up test environment for session state testing."""
        super().setup_method()
        self.config_workflow = ConfigurationWorkflowComponent(self)
        self.training_workflow = TrainingWorkflowComponent(self)

    def test_session_persistence_across_page_navigation(self) -> None:
        """Test session state persistence during page navigation."""
        session_proxy = self.create_session_state_proxy(
            {
                "config_path": "test_config.yaml",
                "theme": "dark",
                "current_page": "Config",
            }
        )

        # Simulate state transitions for page navigation
        transitions = [
            {"current_page": "Architecture"},
            {"current_page": "Training"},
            {"current_page": "Results"},
            {"current_page": "Config"},
        ]

        result = self.simulate_session_state_persistence(
            session_proxy, transitions
        )

        assert result["transitions_applied"] == 4
        assert result["transitions_successful"] == 4
        assert len(result["persistence_violations"]) == 0
        assert result["final_state"]["config_path"] == "test_config.yaml"
        assert result["final_state"]["theme"] == "dark"

    def test_config_to_training_transition(self) -> None:
        """Test session state transitions from config to training."""
        session_proxy = self.create_session_state_proxy()

        result = self.test_session_state_transitions(
            session_proxy, "config_to_training"
        )

        assert result["success"]
        assert "config_loaded" in result["actual_transitions"]
        assert "page_change" in result["actual_transitions"]
        assert "training_ready" in result["actual_transitions"]
        assert session_proxy["current_page"] == "Training"

    def test_training_to_results_transition(self) -> None:
        """Test session state transitions from training to results."""
        session_proxy = self.create_session_state_proxy(
            {
                "training_active": True,
                "current_page": "Training",
            }
        )

        result = self.test_session_state_transitions(
            session_proxy, "training_to_results"
        )

        assert result["success"]
        assert "training_complete" in result["actual_transitions"]
        assert "results_available" in result["actual_transitions"]
        assert "page_change" in result["actual_transitions"]
        assert not session_proxy["training_active"]
        assert session_proxy["current_page"] == "Results"

    def test_error_recovery_state_management(self) -> None:
        """Test session state during error conditions and recovery."""
        session_proxy = self.create_session_state_proxy(
            {
                "config_path": "important_config.yaml",
                "training_active": False,
            }
        )

        result = self.test_session_state_transitions(
            session_proxy, "error_recovery"
        )

        assert result["success"]
        assert "error_occurred" in result["actual_transitions"]
        assert "state_preserved" in result["actual_transitions"]
        assert "recovery_initiated" in result["actual_transitions"]
        assert session_proxy["config_path"] == "important_config.yaml"
        assert not session_proxy.get("error_state", False)

    def test_session_timeout_and_cleanup(self) -> None:
        """Test session timeout handling and state cleanup."""
        session_proxy = self.create_session_state_proxy(
            {
                "training_active": True,
                "training_progress": 0.5,
                "error_state": False,
                "config_path": "persistent_config.yaml",
            }
        )

        result = self.test_session_state_transitions(
            session_proxy, "session_timeout"
        )

        assert result["success"]
        assert "timeout_detected" in result["actual_transitions"]
        assert "cleanup_executed" in result["actual_transitions"]
        assert "state_reset" in result["actual_transitions"]

        # Volatile state should be cleaned up
        assert "training_active" not in session_proxy
        assert "training_progress" not in session_proxy
        assert "error_state" not in session_proxy

        # Persistent config should remain
        assert session_proxy["config_path"] == "persistent_config.yaml"

    def test_session_state_lifecycle_complete(self) -> None:
        """Test complete session state lifecycle management."""

        def create_function(proxy: dict[str, Any]) -> None:
            """Create initial session state."""
            proxy["initialized"] = True
            proxy["creation_time"] = "test_time"

        def use_function(proxy: dict[str, Any]) -> None:
            """Use session state."""
            proxy["used"] = True
            proxy["operations_count"] = proxy.get("operations_count", 0) + 1

        def cleanup_function(proxy: dict[str, Any]) -> None:
            """Cleanup session state."""
            proxy["cleaned_up"] = True
            if "temporary_data" in proxy:
                del proxy["temporary_data"]

        result = self.validate_session_state_lifecycle(
            create_function, use_function, cleanup_function
        )

        assert result["creation_successful"]
        assert result["usage_successful"]
        assert result["cleanup_successful"]
        assert len(result["lifecycle_errors"]) == 0

        # Verify state progression
        after_creation = result["state_snapshots"]["after_creation"]
        assert after_creation["initialized"]
        assert after_creation["creation_time"] == "test_time"

        after_usage = result["state_snapshots"]["after_usage"]
        assert after_usage["used"]
        assert after_usage["operations_count"] == 1

        after_cleanup = result["state_snapshots"]["after_cleanup"]
        assert after_cleanup["cleaned_up"]

    def test_session_state_snapshot_comparison(self) -> None:
        """Test session state snapshot creation and comparison."""
        session_proxy = self.create_session_state_proxy(
            {
                "test_key": "test_value",
                "number_key": 42,
            }
        )

        snapshot1 = self.create_session_state_snapshot(session_proxy)
        assert "timestamp" in snapshot1
        assert snapshot1["state_keys"] == ["test_key", "number_key"]

        # Modify session state
        session_proxy["test_key"] = "modified_value"
        session_proxy["new_key"] = "new_value"
        del session_proxy["number_key"]

        snapshot2 = self.create_session_state_snapshot(session_proxy)
        comparison = self.compare_session_state_snapshots(snapshot1, snapshot2)

        assert not comparison["snapshots_equal"]
        assert "new_key" in comparison["keys_added"]
        assert "number_key" in comparison["keys_removed"]
        assert "test_key" in comparison["keys_modified"]
        assert comparison["time_difference"] > 0

    def test_cross_component_state_sharing(self) -> None:
        """Test session state sharing between workflow components."""
        # Create a temporary config file
        config_path = self.temp_path / "test_config.yaml"
        config_content = """
        model:
          encoder: resnet50
          decoder: unet
        training:
          epochs: 10
          batch_size: 16
        """
        config_path.write_text(config_content)

        # Test configuration workflow
        config_result = self.config_workflow.execute_config_loading_workflow(
            config_path
        )
        assert config_result["success"]

        # Create session proxy with config results
        session_proxy = self.create_session_state_proxy(
            {
                "config_path": str(config_path),
                "config_loaded": config_result["config_loaded"],
            }
        )

        # Verify state sharing works
        assert session_proxy["config_path"] == str(config_path)
        assert session_proxy["config_loaded"]

        # Test training workflow can access shared state
        training_dir = self.temp_path / "training_run"
        training_dir.mkdir(exist_ok=True)
        training_result = (
            self.training_workflow.execute_training_setup_workflow(
                config={"training": {"epochs": 10, "learning_rate": 0.001}},
                run_directory=training_dir,
            )
        )

        # Update session with training state
        session_proxy.update(
            {
                "training_ready": training_result["ready_for_training"],
                "run_directory": str(training_dir),
            }
        )

        # Verify cross-component state consistency
        snapshot = self.create_session_state_snapshot(session_proxy)
        assert "config_path" in snapshot["state_keys"]
        assert "training_ready" in snapshot["state_keys"]
        assert "run_directory" in snapshot["state_keys"]

    def test_session_state_performance_impact(self) -> None:
        """Test session state operations performance impact."""
        import time

        large_session_proxy = self.create_session_state_proxy()

        # Create a session with substantial data
        for i in range(100):
            large_session_proxy[f"key_{i}"] = f"value_{i}" * 10

        # Measure snapshot creation time
        start_time = time.time()
        snapshot = self.create_session_state_snapshot(large_session_proxy)
        snapshot_time = time.time() - start_time

        # Verify reasonable performance (should be sub-second)
        assert snapshot_time < 1.0
        assert snapshot["state_size"] > 0
        assert len(snapshot["state_keys"]) == 107  # 100 + 7 default keys

        # Measure state transition simulation time
        transitions = [{"new_key": f"value_{i}"} for i in range(10)]

        start_time = time.time()
        result = self.simulate_session_state_persistence(
            large_session_proxy, transitions
        )
        persistence_time = time.time() - start_time

        assert persistence_time < 2.0  # Should complete within 2 seconds
        assert result["transitions_applied"] == 10
        assert result["transitions_successful"] == 10

    def test_session_state_memory_efficiency(self) -> None:
        """Test session state memory usage and cleanup efficiency."""
        session_proxy = self.create_session_state_proxy()

        # Add substantial data
        for i in range(50):
            session_proxy[f"data_{i}"] = ["item"] * 100

        initial_snapshot = self.create_session_state_snapshot(session_proxy)
        initial_size = initial_snapshot["state_size"]

        # Simulate cleanup transition
        cleanup_transitions = [
            {
                key: None
                for key in session_proxy.keys()
                if key.startswith("data_")
            }
        ]

        result = self.simulate_session_state_persistence(
            session_proxy, cleanup_transitions
        )

        final_snapshot = self.create_session_state_snapshot(session_proxy)
        final_size = final_snapshot["state_size"]

        # Verify significant size reduction
        assert final_size < initial_size * 0.5  # At least 50% reduction
        assert result["transitions_applied"] == 1
        assert len(result["persistence_violations"]) == 0
