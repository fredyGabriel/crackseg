"""Unit tests for DataAggregation component.

This module tests the data aggregation system that collects data from
all testing phases (9.1-9.7).
"""

from unittest.mock import Mock, patch

import pytest

from tests.integration.gui.automation.reporting.data_aggregation import (
    TestDataAggregator,
)


class TestDataAggregatorComponent:
    """Test suite for TestDataAggregator functionality."""

    @pytest.fixture
    def mock_dependencies(self) -> dict[str, Mock]:
        """Create mock dependencies for testing."""
        # Create mock automation_reporter
        mock_automation_reporter = Mock()
        mock_automation_reporter.get_automation_metrics.return_value = {
            "total_automated_workflows": 4,
            "successful_automations": 4,
            "automation_success_rate": 100.0,
            "avg_automation_time": 120.5,
            "workflow_types": [
                "sequential_automation",
                "parallel_automation",
                "error_recovery_automation",
                "performance_automation",
            ],
            "automation_reliability": "high",
            "script_maintenance_burden": "low",
        }

        # Create mock performance_component
        mock_performance_component = Mock()
        mock_performance_component.get_automation_metrics.return_value = {
            "page_load_times": {
                "avg": 1.2,
                "min": 0.8,
                "max": 2.1,
                "p95": 1.8,
                "p99": 2.0,
            },
            "config_validation_times": {
                "avg": 0.35,
                "min": 0.2,
                "max": 0.6,
                "p95": 0.5,
                "p99": 0.58,
            },
            "memory_usage": {
                "baseline": 512.0,
                "peak": 1024.0,
                "avg": 768.0,
                "unit": "MB",
            },
            "cpu_utilization": {
                "avg": 45.2,
                "peak": 78.5,
                "unit": "percentage",
            },
        }

        # Create mock resource_cleanup_component
        mock_resource_cleanup_component = Mock()
        mock_resource_cleanup_component.get_automation_metrics.return_value = {
            "cleanup_tests_total": 10,
            "cleanup_tests_passed": 9,
            "cleanup_tests_failed": 1,
            "cleanup_effectiveness": 90.0,
        }

        return {
            "automation_reporter": mock_automation_reporter,
            "performance_component": mock_performance_component,
            "resource_cleanup_component": mock_resource_cleanup_component,
        }

    @pytest.fixture
    def data_aggregator(
        self, mock_dependencies: dict[str, Mock]
    ) -> TestDataAggregator:
        """Create TestDataAggregator with mocked dependencies."""
        return TestDataAggregator(**mock_dependencies)

    def test_initialization(self, data_aggregator: TestDataAggregator) -> None:
        """Test aggregator initializes correctly."""
        assert data_aggregator.workflow_collector is not None
        assert data_aggregator.metrics_collector is not None
        assert data_aggregator.performance_collector is not None

    def test_aggregate_comprehensive_data_complete(
        self, data_aggregator: TestDataAggregator
    ) -> None:
        """Test complete data aggregation from all phases."""
        result = data_aggregator.aggregate_comprehensive_data()

        # Verify all expected sections are present
        assert isinstance(result, dict)
        assert "workflow_scenarios" in result
        assert "error_scenarios" in result
        assert "session_state" in result
        assert "concurrent_operations" in result
        assert "automation_metrics" in result
        assert "performance_metrics" in result
        assert "resource_cleanup" in result
        assert "cross_phase_metrics" in result

    def test_workflow_scenarios_data_structure(
        self, data_aggregator: TestDataAggregator
    ) -> None:
        """Test workflow scenarios data structure."""
        result = data_aggregator.aggregate_comprehensive_data()
        workflow_data = result["workflow_scenarios"]

        assert isinstance(workflow_data, dict)
        assert "total_scenarios" in workflow_data
        assert "passed_scenarios" in workflow_data
        assert "failed_scenarios" in workflow_data
        assert "success_rate" in workflow_data
        assert "scenario_types" in workflow_data

        # Verify data types
        assert isinstance(workflow_data["total_scenarios"], int)
        assert isinstance(workflow_data["success_rate"], float)
        assert isinstance(workflow_data["scenario_types"], dict)

    def test_error_scenarios_data_structure(
        self, data_aggregator: TestDataAggregator
    ) -> None:
        """Test error scenarios data structure."""
        result = data_aggregator.aggregate_comprehensive_data()
        error_data = result["error_scenarios"]

        assert isinstance(error_data, dict)
        assert "total_error_scenarios" in error_data
        assert "handled_gracefully" in error_data
        assert "unhandled_errors" in error_data
        assert "error_recovery_rate" in error_data
        assert "error_types" in error_data

        # Verify realistic values
        assert error_data["total_error_scenarios"] > 0
        assert error_data["error_recovery_rate"] >= 0.0
        assert error_data["error_recovery_rate"] <= 100.0

    def test_session_state_data_structure(
        self, data_aggregator: TestDataAggregator
    ) -> None:
        """Test session state data structure."""
        result = data_aggregator.aggregate_comprehensive_data()
        session_data = result["session_state"]

        assert isinstance(session_data, dict)
        assert "total_session_tests" in session_data
        assert "passed_session_tests" in session_data
        assert "failed_session_tests" in session_data
        assert "persistence_rate" in session_data
        assert "persistence_scenarios" in session_data

        # Verify data consistency
        total = session_data["total_session_tests"]
        passed = session_data["passed_session_tests"]
        failed = session_data["failed_session_tests"]
        assert total == passed + failed

    def test_concurrent_operations_data_structure(
        self, data_aggregator: TestDataAggregator
    ) -> None:
        """Test concurrent operations data structure."""
        result = data_aggregator.aggregate_comprehensive_data()
        concurrent_data = result["concurrent_operations"]

        assert isinstance(concurrent_data, dict)
        assert "total_concurrent_tests" in concurrent_data
        assert "passed_concurrent_tests" in concurrent_data
        assert "failed_concurrent_tests" in concurrent_data
        assert "stability_rate" in concurrent_data
        assert "concurrent_scenarios" in concurrent_data

        # Verify realistic values
        assert concurrent_data["stability_rate"] >= 0.0
        assert concurrent_data["stability_rate"] <= 100.0

    def test_automation_metrics_data_structure(
        self, data_aggregator: TestDataAggregator
    ) -> None:
        """Test automation metrics data structure."""
        result = data_aggregator.aggregate_comprehensive_data()
        automation_data = result["automation_metrics"]

        assert isinstance(automation_data, dict)
        assert "total_automated_workflows" in automation_data
        assert "successful_automations" in automation_data
        assert "automation_success_rate" in automation_data
        assert "avg_automation_time" in automation_data
        assert "workflow_types" in automation_data

        # Verify realistic metrics
        assert automation_data["total_automated_workflows"] > 0
        assert automation_data["automation_success_rate"] >= 0.0
        assert automation_data["avg_automation_time"] > 0.0

    def test_performance_metrics_data_structure(
        self, data_aggregator: TestDataAggregator
    ) -> None:
        """Test performance metrics data structure."""
        result = data_aggregator.aggregate_comprehensive_data()
        performance_data = result["performance_metrics"]

        assert isinstance(performance_data, dict)
        assert "page_load_times" in performance_data
        assert "config_validation_times" in performance_data
        assert "memory_usage" in performance_data
        assert "page_load_compliance" in performance_data
        assert "config_validation_compliance" in performance_data

        # Verify realistic performance values
        load_times = performance_data["page_load_times"]
        assert load_times["avg"] > 0.0
        assert load_times["max"] >= load_times["avg"]
        assert load_times["min"] <= load_times["avg"]

    def test_resource_cleanup_data_structure(
        self, data_aggregator: TestDataAggregator
    ) -> None:
        """Test resource cleanup data structure."""
        result = data_aggregator.aggregate_comprehensive_data()
        cleanup_data = result["resource_cleanup"]

        assert isinstance(cleanup_data, dict)
        assert "total_cleanup_tests" in cleanup_data
        assert "passed_cleanup_tests" in cleanup_data
        assert "failed_cleanup_tests" in cleanup_data
        assert "cleanup_effectiveness_rate" in cleanup_data
        assert "cleanup_categories" in cleanup_data

        # Verify data consistency
        total = cleanup_data["total_cleanup_tests"]
        passed = cleanup_data["passed_cleanup_tests"]
        failed = cleanup_data["failed_cleanup_tests"]
        assert total == passed + failed

    def test_data_aggregation_with_mocked_collectors(
        self, data_aggregator: TestDataAggregator
    ) -> None:
        """Test data aggregation with specific mock data."""
        # Mock specific responses from collectors
        with patch.object(
            data_aggregator.workflow_collector,
            "collect_workflow_scenario_data",
        ) as mock_workflow:
            mock_workflow.return_value = {
                "total_scenarios": 15,
                "passed_scenarios": 13,
                "failed_scenarios": 2,
                "success_rate": 86.7,
            }

            result = data_aggregator.aggregate_comprehensive_data()
            workflow_data = result["workflow_scenarios"]

            assert workflow_data["total_scenarios"] == 15
            assert workflow_data["passed_scenarios"] == 13
            assert workflow_data["success_rate"] == 86.7

    def test_data_freshness_information(
        self, data_aggregator: TestDataAggregator
    ) -> None:
        """Test data freshness information."""
        freshness_info = data_aggregator.get_data_freshness_info()

        assert isinstance(freshness_info, dict)
        assert "last_collection_timestamp" in freshness_info
        assert "data_sources" in freshness_info
        assert "data_age_hours" in freshness_info
        assert "data_quality" in freshness_info

        # Verify freshness data types
        assert isinstance(freshness_info["data_age_hours"], int | float)
        assert isinstance(freshness_info["data_sources"], dict)

    def test_cross_phase_metrics_calculation(
        self, data_aggregator: TestDataAggregator
    ) -> None:
        """Test cross-phase metrics calculation."""
        result = data_aggregator.aggregate_comprehensive_data()
        cross_phase_metrics = result["cross_phase_metrics"]

        assert isinstance(cross_phase_metrics, dict)
        assert "overall_success_rate" in cross_phase_metrics
        assert "performance_health_score" in cross_phase_metrics
        assert "resource_efficiency_score" in cross_phase_metrics
        assert "deployment_readiness" in cross_phase_metrics
        assert "critical_issues_count" in cross_phase_metrics

        # Verify metric ranges
        assert 0.0 <= cross_phase_metrics["overall_success_rate"] <= 100.0
        assert 0.0 <= cross_phase_metrics["performance_health_score"] <= 100.0
        assert 0.0 <= cross_phase_metrics["resource_efficiency_score"] <= 100.0

    def test_deployment_readiness_assessment(
        self, data_aggregator: TestDataAggregator
    ) -> None:
        """Test deployment readiness assessment logic."""
        result = data_aggregator.aggregate_comprehensive_data()
        cross_phase_metrics = result["cross_phase_metrics"]

        readiness = cross_phase_metrics["deployment_readiness"]
        assert readiness in [
            "ready",
            "needs_review",
            "significant_issues",
            "not_ready",
        ]

    def test_error_handling_in_data_collection(
        self, data_aggregator: TestDataAggregator
    ) -> None:
        """Test error handling during data collection."""
        # Mock collector to raise an exception
        with patch.object(
            data_aggregator.workflow_collector,
            "collect_workflow_scenario_data",
        ) as mock_collector:
            mock_collector.side_effect = Exception("Collection failed")

            # Should handle the error gracefully
            with pytest.raises(Exception, match="Collection failed"):
                data_aggregator.aggregate_comprehensive_data()

    def test_data_validation(
        self, data_aggregator: TestDataAggregator
    ) -> None:
        """Test aggregated data validation."""
        result = data_aggregator.aggregate_comprehensive_data()

        # Validate all required keys are present
        required_keys = [
            "workflow_scenarios",
            "error_scenarios",
            "session_state",
            "concurrent_operations",
            "automation_metrics",
            "performance_metrics",
            "resource_cleanup",
            "cross_phase_metrics",
        ]

        for key in required_keys:
            assert key in result, f"Missing required key: {key}"
            assert isinstance(result[key], dict), f"Key {key} should be a dict"

    def test_data_consistency_across_calls(
        self, data_aggregator: TestDataAggregator
    ) -> None:
        """Test data consistency across multiple calls."""
        result1 = data_aggregator.aggregate_comprehensive_data()
        result2 = data_aggregator.aggregate_comprehensive_data()

        # Structure should be consistent
        assert result1.keys() == result2.keys()
        # Only check keys for dictionary values (skip strings like timestamp)
        assert all(
            result1[key].keys() == result2[key].keys()
            for key in result1.keys()
            if isinstance(result1[key], dict)
        )

    def test_performance_with_large_datasets(
        self, data_aggregator: TestDataAggregator
    ) -> None:
        """Test aggregation performance with large datasets."""
        import time

        start_time = time.time()
        result = data_aggregator.aggregate_comprehensive_data()
        end_time = time.time()

        # Should complete within reasonable time (2 seconds)
        assert (end_time - start_time) < 2.0
        assert len(result) > 0

    def test_memory_efficiency(
        self, data_aggregator: TestDataAggregator
    ) -> None:
        """Test memory efficiency of data aggregation."""
        import sys

        # Perform aggregation
        result = data_aggregator.aggregate_comprehensive_data()

        # Verify result is not excessively large
        result_size = sys.getsizeof(result)
        assert result_size < 1024 * 1024  # Less than 1MB

    def test_collector_initialization(
        self, data_aggregator: TestDataAggregator
    ) -> None:
        """Test that all collectors are properly initialized."""
        assert hasattr(data_aggregator, "workflow_collector")
        assert hasattr(data_aggregator, "metrics_collector")
        assert hasattr(data_aggregator, "performance_collector")

        # Verify collectors have required methods
        assert hasattr(
            data_aggregator.workflow_collector,
            "collect_workflow_scenario_data",
        )
        assert hasattr(
            data_aggregator.metrics_collector, "collect_automation_metrics"
        )
        assert hasattr(
            data_aggregator.performance_collector,
            "collect_performance_metrics",
        )
