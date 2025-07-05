"""Performance integration for parallel test execution framework.

This module provides integration between the existing PerformanceMonitor
and the parallel test execution framework, enabling performance monitoring
across multiple test workers and consolidated reporting.
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tests.e2e.config.resource_manager import (
    ResourceManager,
    global_resource_manager,
)
from tests.e2e.helpers.performance_monitoring import (
    ExtendedMemoryMetrics,
    PageLoadMetrics,
    PerformanceMetric,
    PerformanceMonitor,
    PerformanceReport,
)

logger = logging.getLogger(__name__)


@dataclass
class WorkerPerformanceData:
    """Performance data from a single test worker."""

    worker_id: str
    test_name: str
    performance_report: PerformanceReport
    start_time: float
    end_time: float
    success: bool = True
    error_message: str | None = None

    @property
    def duration(self) -> float:
        """Get worker execution duration."""
        return self.end_time - self.start_time


@dataclass
class ParallelPerformanceReport:
    """Consolidated performance report from parallel test execution."""

    test_suite_name: str
    start_time: float
    end_time: float
    total_workers: int
    successful_workers: int
    worker_reports: list[WorkerPerformanceData] = field(default_factory=list)
    aggregated_metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def total_duration(self) -> float:
        """Get total parallel execution duration."""
        return self.end_time - self.start_time

    @property
    def average_worker_duration(self) -> float:
        """Get average worker execution time."""
        if not self.worker_reports:
            return 0.0

        total_time = sum(report.duration for report in self.worker_reports)
        return total_time / len(self.worker_reports)

    @property
    def efficiency_ratio(self) -> float:
        """Calculate parallel execution efficiency (0.0 to 1.0)."""
        if not self.worker_reports or self.total_duration == 0:
            return 0.0

        sequential_time = sum(
            report.duration for report in self.worker_reports
        )
        return min(
            1.0, sequential_time / (self.total_duration * self.total_workers)
        )


class ParallelPerformanceMonitor:
    """Performance monitor for parallel test execution."""

    def __init__(self, test_suite_name: str) -> None:
        """Initialize parallel performance monitor.

        Args:
            test_suite_name: Name of the test suite being executed
        """
        self.test_suite_name = test_suite_name
        self.worker_monitors: dict[str, PerformanceMonitor] = {}
        self.worker_data: dict[str, WorkerPerformanceData] = {}
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.monitoring_active = False
        self._lock = threading.Lock()
        self.logger = logging.getLogger(f"{__name__}.{test_suite_name}")

    def start_suite_monitoring(self) -> None:
        """Start monitoring for the entire test suite."""
        with self._lock:
            self.start_time = time.time()
            self.monitoring_active = True
            self.worker_monitors.clear()
            self.worker_data.clear()

        self.logger.info(
            f"Parallel performance monitoring started for "
            f"{self.test_suite_name}"
        )

    def stop_suite_monitoring(self) -> None:
        """Stop monitoring for the entire test suite."""
        with self._lock:
            self.end_time = time.time()
            self.monitoring_active = False

            # Stop all active worker monitors
            for monitor in self.worker_monitors.values():
                if monitor.monitoring_active:
                    monitor.stop_monitoring()

        self.logger.info(
            f"Parallel performance monitoring stopped for "
            f"{self.test_suite_name}"
        )

    def create_worker_monitor(
        self, worker_id: str, test_name: str
    ) -> PerformanceMonitor:
        """Create and register a performance monitor for a specific worker.

        Args:
            worker_id: Unique identifier for the worker
            test_name: Name of the test being executed

        Returns:
            PerformanceMonitor instance for the worker
        """
        with self._lock:
            monitor_name = f"{self.test_suite_name}.{worker_id}.{test_name}"
            monitor = PerformanceMonitor(monitor_name)
            self.worker_monitors[worker_id] = monitor

        self.logger.debug(
            f"Created performance monitor for worker {worker_id}"
        )
        return monitor

    def register_worker_completion(
        self,
        worker_id: str,
        test_name: str,
        success: bool = True,
        error_message: str | None = None,
    ) -> None:
        """Register completion of a worker's test execution.

        Args:
            worker_id: Unique identifier for the worker
            test_name: Name of the completed test
            success: Whether the test completed successfully
            error_message: Optional error message if test failed
        """
        with self._lock:
            monitor = self.worker_monitors.get(worker_id)
            if not monitor:
                self.logger.warning(f"No monitor found for worker {worker_id}")
                return

            # Ensure monitor is stopped
            if monitor.monitoring_active:
                monitor.stop_monitoring()

            # Create worker performance data
            worker_data = WorkerPerformanceData(
                worker_id=worker_id,
                test_name=test_name,
                performance_report=monitor.report,
                start_time=monitor.report.start_time,
                end_time=monitor.report.end_time,
                success=success,
                error_message=error_message,
            )

            self.worker_data[worker_id] = worker_data

        self.logger.info(
            f"Registered completion for worker {worker_id}: "
            f"{'SUCCESS' if success else 'FAILED'} in "
            f"{worker_data.duration:.3f}s"
        )

    def add_suite_metric(
        self,
        metric_name: str,
        value: float,
        unit: str = "ms",
        context: dict[str, Any] | None = None,
    ) -> None:
        """Add a custom metric at the test suite level.

        Args:
            metric_name: Name of the metric
            value: Measured value
            unit: Unit of measurement
            context: Additional context information
        """
        metric: PerformanceMetric = {
            "metric_name": metric_name,
            "value": value,
            "unit": unit,
            "timestamp": time.time(),
            "context": context or {},
        }

        # Store suite-level metrics separately
        if "suite_metrics" not in self.worker_data:
            self.worker_data["suite_metrics"] = {"metrics": []}

        self.worker_data["suite_metrics"]["metrics"].append(metric)
        self.logger.info(f"Suite metric added: {metric_name} = {value} {unit}")

    def generate_consolidated_report(
        self, output_file: Path | str | None = None
    ) -> ParallelPerformanceReport:
        """Generate consolidated performance report from all workers.

        Args:
            output_file: Optional file path to save report

        Returns:
            Consolidated parallel performance report
        """
        if self.monitoring_active:
            self.stop_suite_monitoring()

        # Create base report
        report = ParallelPerformanceReport(
            test_suite_name=self.test_suite_name,
            start_time=self.start_time,
            end_time=self.end_time,
            total_workers=len(self.worker_data),
            successful_workers=sum(
                1
                for data in self.worker_data.values()
                if isinstance(data, WorkerPerformanceData) and data.success
            ),
            worker_reports=[
                data
                for data in self.worker_data.values()
                if isinstance(data, WorkerPerformanceData)
            ],
        )

        # Aggregate metrics across workers
        report.aggregated_metrics = self._aggregate_worker_metrics()

        # Save report if requested
        if output_file:
            self._save_consolidated_report(report, output_file)

        return report

    def _aggregate_worker_metrics(self) -> dict[str, Any]:
        """Aggregate performance metrics across all workers."""
        aggregated: dict[str, Any] = {
            "execution_summary": {},
            "performance_totals": {},
            "resource_usage": {},
            "parallel_efficiency": {},
            "worker_breakdown": {},
        }

        worker_reports = [
            data
            for data in self.worker_data.values()
            if isinstance(data, WorkerPerformanceData)
        ]

        if not worker_reports:
            return aggregated

        # Execution summary
        total_tests = len(worker_reports)
        successful_tests = sum(
            1 for report in worker_reports if report.success
        )

        aggregated["execution_summary"] = {
            "total_workers": total_tests,
            "successful_workers": successful_tests,
            "failed_workers": total_tests - successful_tests,
            "success_rate": (
                successful_tests / total_tests if total_tests > 0 else 0
            ),
            "total_execution_time": self.total_duration,
            "average_worker_time": sum(
                report.duration for report in worker_reports
            )
            / total_tests,
        }

        # Performance totals
        all_page_loads: list[PageLoadMetrics] = []
        all_memory_snapshots: list[ExtendedMemoryMetrics] = []
        all_interactions: list[dict[str, Any]] = []

        for report in worker_reports:
            all_page_loads.extend(report.performance_report.page_loads)
            all_memory_snapshots.extend(
                report.performance_report.memory_snapshots
            )
            all_interactions.extend(report.performance_report.interactions)

        aggregated["performance_totals"] = {
            "total_page_loads": len(all_page_loads),
            "average_page_load_time": (
                sum(pl["load_complete"] for pl in all_page_loads)
                / len(all_page_loads)
                if all_page_loads
                else 0.0
            ),
            "total_interactions": len(all_interactions),
            "successful_interactions": sum(
                1 for i in all_interactions if i.get("success", False)
            ),
            "total_memory_snapshots": len(all_memory_snapshots),
        }

        # Resource usage
        if all_memory_snapshots:
            peak_memory = max(
                snapshot["rss_memory_mb"] for snapshot in all_memory_snapshots
            )
            avg_cpu = sum(
                snapshot["cpu_percent"] for snapshot in all_memory_snapshots
            ) / len(all_memory_snapshots)

            aggregated["resource_usage"] = {
                "peak_memory_mb": peak_memory,
                "average_cpu_percent": avg_cpu,
                "total_memory_samples": len(all_memory_snapshots),
            }

        # Parallel efficiency
        sequential_time = sum(report.duration for report in worker_reports)
        parallel_time = self.total_duration

        aggregated["parallel_efficiency"] = {
            "sequential_time_estimate": sequential_time,
            "actual_parallel_time": parallel_time,
            "speedup_ratio": (
                sequential_time / parallel_time if parallel_time > 0 else 0
            ),
            "efficiency_percentage": (
                min(
                    100,
                    (sequential_time / (parallel_time * total_tests)) * 100,
                )
                if parallel_time > 0 and total_tests > 0
                else 0
            ),
        }

        # Worker breakdown
        aggregated["worker_breakdown"] = {
            worker_data.worker_id: {
                "test_name": worker_data.test_name,
                "duration": worker_data.duration,
                "success": worker_data.success,
                "page_loads": len(worker_data.performance_report.page_loads),
                "memory_snapshots": len(
                    worker_data.performance_report.memory_snapshots
                ),
                "interactions": len(
                    worker_data.performance_report.interactions
                ),
                "peak_memory_mb": (
                    worker_data.performance_report.peak_memory_usage
                ),
                "error": worker_data.error_message,
            }
            for worker_data in worker_reports
        }

        return aggregated

    def _save_consolidated_report(
        self, report: ParallelPerformanceReport, output_file: Path | str
    ) -> None:
        """Save consolidated report to file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to JSON-serializable format
        report_data = {
            "test_suite_name": report.test_suite_name,
            "execution_window": {
                "start_time": report.start_time,
                "end_time": report.end_time,
                "total_duration": report.total_duration,
            },
            "summary": {
                "total_workers": report.total_workers,
                "successful_workers": report.successful_workers,
                "average_worker_duration": report.average_worker_duration,
                "efficiency_ratio": report.efficiency_ratio,
            },
            "aggregated_metrics": report.aggregated_metrics,
            "individual_workers": [
                {
                    "worker_id": worker.worker_id,
                    "test_name": worker.test_name,
                    "duration": worker.duration,
                    "success": worker.success,
                    "error_message": worker.error_message,
                    "performance_summary": {
                        "page_loads": len(
                            worker.performance_report.page_loads
                        ),
                        "memory_snapshots": len(
                            worker.performance_report.memory_snapshots
                        ),
                        "interactions": len(
                            worker.performance_report.interactions
                        ),
                        "peak_memory_mb": (
                            worker.performance_report.peak_memory_usage
                        ),
                    },
                }
                for worker in report.worker_reports
            ],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, default=str)

        self.logger.info(
            f"Consolidated performance report saved to {output_path}"
        )


class ParallelPerformanceIntegration:
    """Main integration class for parallel performance monitoring."""

    def __init__(
        self, resource_manager: ResourceManager | None = None
    ) -> None:
        """Initialize parallel performance integration.

        Args:
            resource_manager: Optional resource manager for coordination
        """
        self.resource_manager = resource_manager or global_resource_manager
        self.suite_monitors: dict[str, ParallelPerformanceMonitor] = {}
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)

    def create_suite_monitor(
        self, test_suite_name: str
    ) -> ParallelPerformanceMonitor:
        """Create a performance monitor for a test suite.

        Args:
            test_suite_name: Name of the test suite

        Returns:
            ParallelPerformanceMonitor instance
        """
        with self._lock:
            if test_suite_name in self.suite_monitors:
                self.logger.warning(
                    f"Suite monitor for {test_suite_name} already exists"
                )
                return self.suite_monitors[test_suite_name]

            monitor = ParallelPerformanceMonitor(test_suite_name)
            self.suite_monitors[test_suite_name] = monitor

        self.logger.info(
            f"Created parallel performance monitor for suite: "
            f"{test_suite_name}"
        )
        return monitor

    def get_suite_monitor(
        self, test_suite_name: str
    ) -> ParallelPerformanceMonitor | None:
        """Get existing suite monitor by name.

        Args:
            test_suite_name: Name of the test suite

        Returns:
            ParallelPerformanceMonitor instance or None if not found
        """
        return self.suite_monitors.get(test_suite_name)

    def cleanup_suite_monitor(self, test_suite_name: str) -> None:
        """Clean up a suite monitor after completion.

        Args:
            test_suite_name: Name of the test suite
        """
        with self._lock:
            monitor = self.suite_monitors.pop(test_suite_name, None)
            if monitor and monitor.monitoring_active:
                monitor.stop_suite_monitoring()

        self.logger.info(f"Cleaned up suite monitor: {test_suite_name}")

    def generate_suite_report(
        self, test_suite_name: str, output_file: Path | str | None = None
    ) -> ParallelPerformanceReport | None:
        """Generate performance report for a test suite.

        Args:
            test_suite_name: Name of the test suite
            output_file: Optional file path to save report

        Returns:
            ParallelPerformanceReport or None if suite not found
        """
        monitor = self.get_suite_monitor(test_suite_name)
        if not monitor:
            self.logger.error(f"No monitor found for suite: {test_suite_name}")
            return None

        return monitor.generate_consolidated_report(output_file)


# Global integration instance for easy access
global_performance_integration = ParallelPerformanceIntegration()


# Utility functions for common usage patterns
def create_worker_performance_monitor(
    suite_name: str, worker_id: str, test_name: str
) -> PerformanceMonitor:
    """Create a performance monitor for a test worker.

    Args:
        suite_name: Name of the test suite
        worker_id: Unique identifier for the worker
        test_name: Name of the test being executed

    Returns:
        PerformanceMonitor instance for the worker
    """
    suite_monitor = global_performance_integration.get_suite_monitor(
        suite_name
    )
    if not suite_monitor:
        suite_monitor = global_performance_integration.create_suite_monitor(
            suite_name
        )
        suite_monitor.start_suite_monitoring()

    return suite_monitor.create_worker_monitor(worker_id, test_name)


def register_worker_performance_completion(
    suite_name: str,
    worker_id: str,
    test_name: str,
    success: bool = True,
    error_message: str | None = None,
) -> None:
    """Register completion of a worker's performance monitoring.

    Args:
        suite_name: Name of the test suite
        worker_id: Unique identifier for the worker
        test_name: Name of the completed test
        success: Whether the test completed successfully
        error_message: Optional error message if test failed
    """
    suite_monitor = global_performance_integration.get_suite_monitor(
        suite_name
    )
    if suite_monitor:
        suite_monitor.register_worker_completion(
            worker_id, test_name, success, error_message
        )


def generate_parallel_performance_report(
    suite_name: str, output_file: Path | str | None = None
) -> ParallelPerformanceReport | None:
    """Generate consolidated performance report for a test suite.

    Args:
        suite_name: Name of the test suite
        output_file: Optional file path to save report

    Returns:
        ParallelPerformanceReport or None if suite not found
    """
    return global_performance_integration.generate_suite_report(
        suite_name, output_file
    )
