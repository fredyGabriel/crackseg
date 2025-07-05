"""System stability concurrent operation tests.

Tests focused on system stability, performance monitoring, and load testing
under concurrent operations. Extracted from oversized test_concurrent_operations.py.
"""


from ..test_base import WorkflowTestBase
from ..workflow_components.concurrent import ConcurrentOperationMixin
from ..workflow_components.config_workflow import (
    ConfigurationWorkflowComponent,
)
from ..workflow_components.error_scenario_mixin import ErrorScenarioMixin
from ..workflow_components.session_state_mixin import SessionStateMixin
from ..workflow_components.training_workflow import TrainingWorkflowComponent


class TestSystemStability(
    WorkflowTestBase,
    ConcurrentOperationMixin,
    ErrorScenarioMixin,
    SessionStateMixin,
):
    """Test system stability under concurrent operations and load."""

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

    def test_system_stability_under_concurrent_load(self) -> None:
        """Test system stability under various concurrent load scenarios."""
        # Define load scenarios for stability testing
        load_scenarios = [
            {"type": "cpu_intensive", "duration": 0.5},
            {"type": "memory_intensive", "duration": 0.3},
            {"type": "io_intensive", "duration": 0.4},
            {"type": "cpu_intensive", "duration": 0.2},
        ]

        result = self.test_system_stability_under_load(
            load_scenarios=load_scenarios,
            monitoring_duration=2.0,
        )

        # Verify system stability
        assert result[
            "success"
        ], f"System stability failed: {result['stability_violations']}"

        # Verify reasonable resource usage
        metrics = result["system_metrics"]
        assert metrics["max_threads"] < 50, "Too many threads created"
        assert metrics["error_count"] == 0, "Errors occurred during load test"

        if metrics["response_times"]:
            avg_response = metrics["avg_response_time"]
            assert (
                avg_response < 0.5
            ), f"Average response time too high: {avg_response:.3f}s"

    def test_memory_usage_concurrent_operations(self) -> None:
        """Test memory usage patterns during concurrent operations."""
        # Define memory-intensive scenarios
        memory_scenarios = [
            {"type": "memory_intensive", "duration": 0.3},
            {"type": "memory_intensive", "duration": 0.4},
            {"type": "cpu_intensive", "duration": 0.2},
        ]

        result = self.test_system_stability_under_load(
            load_scenarios=memory_scenarios,
            monitoring_duration=1.5,
        )

        # Verify memory usage is within reasonable bounds
        assert result[
            "success"
        ], f"Memory usage test failed: {result['stability_violations']}"

        # Check system metrics
        metrics = result["system_metrics"]
        assert metrics["error_count"] == 0, "Memory errors occurred"

        # Verify response times under memory load
        if metrics["response_times"]:
            max_response = metrics["max_response_time"]
            assert (
                max_response < 1.0
            ), f"Memory operations too slow: {max_response:.3f}s"

    def test_concurrent_load_scalability(self) -> None:
        """Test system scalability under increasing concurrent load."""
        # Test scalability with different load levels
        load_levels = [2, 4, 6]  # Increasing number of concurrent operations

        for load_level in load_levels:
            load_scenarios = [
                {"type": "cpu_intensive", "duration": 0.2}
                for _ in range(load_level)
            ]

            result = self.test_system_stability_under_load(
                load_scenarios=load_scenarios,
                monitoring_duration=1.0,
            )

            # Verify system handles increased load
            assert result["success"], (
                f"Load level {load_level} failed: "
                f"{result['stability_violations']}"
            )

            metrics = result["system_metrics"]

            # Check that response times don't degrade too much
            if metrics["response_times"]:
                avg_response = metrics["avg_response_time"]
                expected_threshold = (
                    0.1 * load_level
                )  # Scale threshold with load

                assert (
                    avg_response < expected_threshold
                ), f"Load level {load_level} too slow: {avg_response:.3f}s"

    def test_error_handling_under_load(self) -> None:
        """Test error handling mechanisms under concurrent load."""
        # Mix of normal and error-prone operations
        mixed_scenarios = [
            {"type": "cpu_intensive", "duration": 0.2},
            {
                "type": "error_prone",
                "duration": 0.1,
            },  # Simulated error scenario
            {"type": "memory_intensive", "duration": 0.3},
            {"type": "cpu_intensive", "duration": 0.2},
        ]

        result = self.test_system_stability_under_load(
            load_scenarios=mixed_scenarios,
            monitoring_duration=1.5,
        )

        # Verify system stability even with errors
        # Allow some violations due to intentional errors
        metrics = result["system_metrics"]

        # Check that system doesn't crash completely
        assert (
            len(metrics["response_times"]) >= 2
        ), "System completely failed under error conditions"

        # Verify that successful operations still complete reasonably fast
        if metrics["response_times"]:
            successful_times = [
                t for t in metrics["response_times"] if t < 1.0
            ]
            assert (
                len(successful_times) >= 2
            ), "Not enough successful operations under error load"

    def test_resource_cleanup_concurrent_operations(self) -> None:
        """Test resource cleanup after concurrent operations."""
        # Run load scenarios and verify cleanup
        cleanup_scenarios = [
            {"type": "memory_intensive", "duration": 0.2},
            {"type": "io_intensive", "duration": 0.3},
            {"type": "cpu_intensive", "duration": 0.1},
        ]

        # Execute load test
        result = self.test_system_stability_under_load(
            load_scenarios=cleanup_scenarios,
            monitoring_duration=1.0,
        )

        # Verify initial execution
        assert (
            len(result["stability_violations"]) <= 1
        ), "Too many stability violations during cleanup test"

        # Perform explicit cleanup
        self.cleanup_concurrent_operations()

        # Run another small load test to verify system is clean
        post_cleanup_scenarios = [
            {"type": "cpu_intensive", "duration": 0.1},
        ]

        post_result = self.test_system_stability_under_load(
            load_scenarios=post_cleanup_scenarios,
            monitoring_duration=0.5,
        )

        # Verify system is clean after cleanup
        assert post_result["success"], (
            f"System not clean after cleanup: "
            f"{post_result['stability_violations']}"
        )

        post_metrics = post_result["system_metrics"]
        assert (
            post_metrics["error_count"] == 0
        ), "Cleanup did not resolve all errors"
