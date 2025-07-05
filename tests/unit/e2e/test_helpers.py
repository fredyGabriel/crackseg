"""Unit tests for E2E helpers implemented in subtask 15.6.

This module tests the helper functions for setup/teardown, API integration,
test coordination, and performance monitoring to ensure they work correctly
in isolation and integrate properly with the existing E2E infrastructure.
"""

import tempfile
import time

import pytest

from tests.e2e.helpers.api_integration import (
    APITestHelper,
    validate_api_responses,
)
from tests.e2e.helpers.performance_monitoring import PerformanceMonitor
from tests.e2e.helpers.setup_teardown import TestEnvironmentManager
from tests.e2e.helpers.test_coordination import TestCoordinator, TestTask


class TestSetupTeardownHelpers:
    """Test suite for setup/teardown helper functions."""

    def test_test_environment_manager_initialization(self) -> None:
        """Test TestEnvironmentManager can be initialized correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = TestEnvironmentManager(
                "test_init", artifacts_base_dir=temp_dir
            )

            assert manager.test_name == "test_init"
            assert manager.state["timestamp"] > 0
            assert isinstance(manager.state["cleanup_paths"], list)
            assert isinstance(manager.state["running_processes"], list)


class TestAPIIntegrationHelpers:
    """Test suite for API integration helper functions."""

    def test_api_test_helper_initialization(self) -> None:
        """Test APITestHelper initialization with different parameters."""
        helper = APITestHelper(base_url="http://localhost:8501", timeout=10.0)

        assert helper.base_url == "http://localhost:8501"
        assert helper.timeout == 10.0

    def test_validate_api_responses(self) -> None:
        """Test API response validation functionality."""
        from tests.e2e.helpers.api_integration import APIResponse

        responses: list[APIResponse] = [
            {
                "status_code": 200,
                "response_time": 0.1,
                "content": {"data": "test"},
                "headers": {},
                "success": True,
                "error": None,
            },
            {
                "status_code": 500,
                "response_time": 0.5,
                "content": None,
                "headers": {},
                "success": False,
                "error": "Server error",
            },
        ]

        validation_report = validate_api_responses(responses)

        assert validation_report["total"] == 2
        assert validation_report["valid"] == 1
        assert validation_report["invalid"] == 1


class TestCoordinationHelpers:
    """Test suite for test coordination helper functions."""

    def test_test_task_initialization(self) -> None:
        """Test TestTask data class initialization."""

        def dummy_test() -> bool:
            return True

        task = TestTask(
            name="test_task",
            test_function=dummy_test,
            dependencies=["dep1", "dep2"],
            priority=1,
            timeout=30.0,
        )

        assert task.name == "test_task"
        assert task.test_function == dummy_test
        assert task.dependencies == ["dep1", "dep2"]

    def test_test_coordinator_initialization(self) -> None:
        """Test TestCoordinator initialization."""
        coordinator = TestCoordinator()

        assert len(coordinator.tasks) == 0
        assert len(coordinator.results) == 0


class TestPerformanceMonitoringHelpers:
    """Test suite for performance monitoring helper functions."""

    def test_performance_monitor_initialization(self) -> None:
        """Test PerformanceMonitor initialization."""
        monitor = PerformanceMonitor("test_perf")

        assert monitor.test_name == "test_perf"
        assert not monitor.monitoring_active

    def test_performance_monitor_start_stop(self) -> None:
        """Test performance monitoring start/stop lifecycle."""
        monitor = PerformanceMonitor("test_lifecycle")

        # Test start
        monitor.start_monitoring()
        assert monitor.monitoring_active
        assert monitor.report.start_time > 0

        time.sleep(0.1)  # Small delay

        # Test stop
        monitor.stop_monitoring()
        assert not monitor.monitoring_active
        assert monitor.report.end_time > monitor.report.start_time

    def test_performance_monitor_add_custom_metric(self) -> None:
        """Test custom metric addition functionality."""
        monitor = PerformanceMonitor("test_metrics")

        monitor.add_custom_metric(
            "test_metric", 123.45, "units", {"context": "test"}
        )

        assert len(monitor.report.metrics) == 1
        metric = monitor.report.metrics[0]
        assert metric["metric_name"] == "test_metric"
        assert metric["value"] == 123.45


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
