"""Data aggregation module for integration test reporting.

This module provides data collection and aggregation orchestration from all
testing phases (9.1-9.7) to support comprehensive stakeholder reporting.
"""

from datetime import datetime
from typing import Any

from ..automation_orchestrator import AutomationReporterImpl
from ..performance_benchmarking import PerformanceBenchmarkingComponent
from ..resource_cleanup_validation import ResourceCleanupValidationComponent
from .data_collectors.metrics_collector import MetricsCollector
from .data_collectors.performance_collector import PerformanceCollector
from .data_collectors.workflow_data_collector import WorkflowDataCollector


class TestDataAggregator:
    """Aggregates data from all testing phases for comprehensive reporting."""

    def __init__(
        self,
        automation_reporter: AutomationReporterImpl,
        performance_component: PerformanceBenchmarkingComponent,
        resource_cleanup_component: ResourceCleanupValidationComponent,
    ) -> None:
        """Initialize test data aggregator.

        Args:
            automation_reporter: Automation reporting component
            performance_component: Performance benchmarking component
            resource_cleanup_component: Resource cleanup validation component
        """
        self.automation_reporter = automation_reporter
        self.performance_component = performance_component
        self.resource_cleanup_component = resource_cleanup_component

        # Initialize specialized collectors
        self.workflow_collector = WorkflowDataCollector()
        self.metrics_collector = MetricsCollector(
            automation_reporter, resource_cleanup_component
        )
        self.performance_collector = PerformanceCollector(
            performance_component
        )

    def aggregate_comprehensive_data(self) -> dict[str, Any]:
        """Aggregate data from all testing phases (9.1-9.7).

        Returns:
            Comprehensive aggregated testing data
        """
        # Collect data from all phases using specialized collectors
        aggregated_data = {
            "timestamp": datetime.now().isoformat(),
            "workflow_scenarios": (
                self.workflow_collector.collect_workflow_scenario_data()
            ),
            "error_scenarios": (
                self.workflow_collector.collect_error_scenario_data()
            ),
            "session_state": (
                self.workflow_collector.collect_session_state_data()
            ),
            "concurrent_operations": (
                self.workflow_collector.collect_concurrent_operations_data()
            ),
            "automation_metrics": (
                self.metrics_collector.collect_automation_metrics()
            ),
            "performance_metrics": (
                self.performance_collector.collect_performance_metrics()
            ),
            "resource_cleanup": (
                self.metrics_collector.collect_resource_cleanup_data()
            ),
            "integration_summary": (
                self.performance_collector.generate_integration_summary()
            ),
        }

        # Add cross-phase calculated metrics
        aggregated_data["cross_phase_metrics"] = (
            self.metrics_collector.calculate_cross_phase_metrics(
                aggregated_data
            )
        )

        return aggregated_data

    def get_data_freshness_info(self) -> dict[str, Any]:
        """Get information about data freshness and collection timestamps.

        Returns:
            Data freshness information
        """
        return self.metrics_collector.get_data_freshness_info()
