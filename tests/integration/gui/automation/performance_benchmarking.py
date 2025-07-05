"""Performance benchmarking component for systematic workflow analysis.

This module extends the automation framework from 9.5 to provide comprehensive
performance benchmarking and analysis capabilities across all workflow
components (9.1-9.4), measuring response times, resource utilization, and
identifying bottlenecks to ensure performance requirements are met.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil

from .automation_protocols import (
    AutomationConfiguration,
    AutomationResult,
)
from .workflow_automation import WorkflowAutomationComponent


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for workflow analysis."""

    # Timing metrics
    response_time_ms: float
    page_load_time_ms: float
    config_validation_time_ms: float
    workflow_execution_time_ms: float

    # Resource utilization metrics
    cpu_usage_percent: float
    memory_usage_mb: float
    memory_peak_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float

    # Throughput metrics
    operations_per_second: float
    concurrent_users_supported: int

    # Quality metrics
    success_rate_percent: float
    error_rate_percent: float

    # Performance requirements validation
    meets_page_load_requirement: bool  # <2s requirement
    meets_config_validation_requirement: bool  # <500ms requirement

    # Additional metadata
    timestamp: datetime = field(default_factory=datetime.now)
    test_scenario: str = ""
    workflow_phase: str = ""


@dataclass
class BottleneckAnalysis:
    """Analysis of performance bottlenecks and optimization opportunities."""

    identified_bottlenecks: list[str] = field(default_factory=list)
    critical_path_components: list[str] = field(default_factory=list)
    resource_constraints: dict[str, float] = field(default_factory=dict)
    optimization_recommendations: list[str] = field(default_factory=list)
    performance_regression_detected: bool = False
    baseline_comparison: dict[str, float] = field(default_factory=dict)


class PerformanceInstrumentationMixin:
    """Mixin for adding performance instrumentation to workflow components."""

    def __init__(self) -> None:
        """Initialize performance monitoring."""
        self.performance_data: list[PerformanceMetrics] = []
        self.baseline_metrics: dict[str, float] = {}

    def measure_performance(
        self,
        operation_name: str,
        operation_callable: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Any, PerformanceMetrics]:
        """Measure performance of a specific operation."""
        # Pre-execution measurements
        start_time = time.time()
        start_memory = psutil.virtual_memory().used / (1024 * 1024)  # MB

        # Execute operation
        try:
            result = operation_callable(*args, **kwargs)
            success = True
        except Exception as e:
            result = e
            success = False

        # Post-execution measurements
        end_time = time.time()
        end_memory = psutil.virtual_memory().used / (1024 * 1024)  # MB
        end_cpu = psutil.cpu_percent(interval=None)

        execution_time_ms = (end_time - start_time) * 1000

        # Create performance metrics
        metrics = PerformanceMetrics(
            response_time_ms=execution_time_ms,
            page_load_time_ms=(
                execution_time_ms if "page" in operation_name.lower() else 0.0
            ),
            config_validation_time_ms=(
                execution_time_ms
                if "config" in operation_name.lower()
                else 0.0
            ),
            workflow_execution_time_ms=(
                execution_time_ms
                if "workflow" in operation_name.lower()
                else 0.0
            ),
            cpu_usage_percent=end_cpu,
            memory_usage_mb=end_memory,
            memory_peak_mb=max(start_memory, end_memory),
            disk_io_read_mb=0.0,  # Would need more complex monitoring
            disk_io_write_mb=0.0,
            operations_per_second=(
                1000.0 / execution_time_ms if execution_time_ms > 0 else 0.0
            ),
            concurrent_users_supported=1,  # Single user for this measurement
            success_rate_percent=100.0 if success else 0.0,
            error_rate_percent=0.0 if success else 100.0,
            meets_page_load_requirement=execution_time_ms
            < 2000,  # <2s requirement
            meets_config_validation_requirement=execution_time_ms
            < 500,  # <500ms requirement
            test_scenario=operation_name,
            workflow_phase="measurement",
        )

        self.performance_data.append(metrics)
        return result, metrics


class PerformanceBenchmarkingComponent(PerformanceInstrumentationMixin):
    """Performance benchmarking component extending automation infrastructure.

    Provides systematic performance analysis and bottleneck identification
    across all workflow components (9.1-9.4) with comprehensive metrics
    collection and analysis capabilities.
    """

    def __init__(self, test_utilities: Any) -> None:
        """Initialize performance benchmarking with test utilities."""
        super().__init__()
        self.test_utilities = test_utilities
        self.workflow_automation = WorkflowAutomationComponent(test_utilities)

    def get_workflow_name(self) -> str:
        """Get the name of this performance benchmarking workflow."""
        return "CrackSeg Performance Benchmarking Suite"

    def execute_automated_workflow(
        self, automation_config: dict[str, Any]
    ) -> AutomationResult:
        """Execute comprehensive performance benchmarking workflow."""
        config = AutomationConfiguration(**automation_config)
        start_time = datetime.now()

        # Execute performance benchmarking phases
        workflow_results = self._benchmark_workflow_performance(config)
        resource_results = self._benchmark_resource_utilization(config)
        scalability_results = self._benchmark_scalability_performance(config)
        bottleneck_analysis = self._analyze_performance_bottlenecks()

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        # Aggregate results
        total_tests = (
            len(workflow_results)
            + len(resource_results)
            + len(scalability_results)
        )
        passed_tests = sum(
            1
            for r in workflow_results + resource_results + scalability_results
            if r.success_rate_percent > 80.0
        )

        # Compile performance metrics
        performance_metrics = self._compile_performance_metrics()

        return AutomationResult(
            workflow_name=self.get_workflow_name(),
            success=passed_tests == total_tests,
            start_time=start_time,
            end_time=end_time,
            execution_time_seconds=execution_time,
            test_count=total_tests,
            passed_count=passed_tests,
            failed_count=total_tests - passed_tests,
            error_details=self._extract_error_details(),
            performance_metrics=performance_metrics,
            artifacts_generated=self._generate_performance_artifacts(config),
            metadata={
                "benchmarking_phases": ["workflow", "resource", "scalability"],
                "bottleneck_analysis": bottleneck_analysis.__dict__,
                "performance_requirements_met": (
                    self._validate_performance_requirements()
                ),
            },
        )

    def validate_automation_preconditions(self) -> bool:
        """Validate that performance benchmarking preconditions are met."""
        return (
            self.workflow_automation.validate_automation_preconditions()
            and hasattr(psutil, "virtual_memory")
            and hasattr(psutil, "cpu_percent")
        )

    def get_automation_metrics(self) -> dict[str, float]:
        """Get performance-specific automation metrics."""
        if not self.performance_data:
            return {"no_performance_data": 0.0}

        avg_response_time = sum(
            m.response_time_ms for m in self.performance_data
        ) / len(self.performance_data)
        avg_memory_usage = sum(
            m.memory_usage_mb for m in self.performance_data
        ) / len(self.performance_data)

        return {
            "avg_response_time_ms": avg_response_time,
            "avg_memory_usage_mb": avg_memory_usage,
            "performance_measurements": float(len(self.performance_data)),
            "page_load_compliance_rate": sum(
                1
                for m in self.performance_data
                if m.meets_page_load_requirement
            )
            / len(self.performance_data)
            * 100,
            "config_validation_compliance_rate": sum(
                1
                for m in self.performance_data
                if m.meets_config_validation_requirement
            )
            / len(self.performance_data)
            * 100,
        }

    def _benchmark_workflow_performance(
        self, config: AutomationConfiguration
    ) -> list[PerformanceMetrics]:
        """Benchmark performance of workflow components (9.1-9.4)."""
        results: list[PerformanceMetrics] = []

        # Benchmark configuration workflow (9.1)
        config_result, config_metrics = self.measure_performance(
            "configuration_workflow_execution",
            self.workflow_automation.execute_configuration_automation,
            config,
        )
        results.append(config_metrics)

        # Benchmark training workflow (9.1)
        training_result, training_metrics = self.measure_performance(
            "training_workflow_execution",
            self.workflow_automation.execute_training_automation,
            config,
        )
        results.append(training_metrics)

        # Benchmark concurrent operations (9.4)
        concurrent_result, concurrent_metrics = self.measure_performance(
            "concurrent_workflow_execution",
            self.workflow_automation.execute_concurrent_automation,
            config,
        )
        results.append(concurrent_metrics)

        return results

    def _benchmark_resource_utilization(
        self, config: AutomationConfiguration
    ) -> list[PerformanceMetrics]:
        """Benchmark resource utilization patterns."""
        results: list[PerformanceMetrics] = []

        # Memory usage benchmarking
        memory_result, memory_metrics = self.measure_performance(
            "memory_intensive_operation",
            self._simulate_memory_intensive_workflow,
        )
        results.append(memory_metrics)

        # CPU usage benchmarking
        cpu_result, cpu_metrics = self.measure_performance(
            "cpu_intensive_operation", self._simulate_cpu_intensive_workflow
        )
        results.append(cpu_metrics)

        return results

    def _benchmark_scalability_performance(
        self, config: AutomationConfiguration
    ) -> list[PerformanceMetrics]:
        """Benchmark scalability under increasing load."""
        results: list[PerformanceMetrics] = []

        for user_count in [1, 2, 5, 10]:
            load_result, load_metrics = self.measure_performance(
                f"scalability_test_{user_count}_users",
                self._simulate_concurrent_users,
                user_count,
            )
            load_metrics.concurrent_users_supported = user_count
            results.append(load_metrics)

        return results

    def _analyze_performance_bottlenecks(self) -> BottleneckAnalysis:
        """Analyze performance data to identify bottlenecks."""
        if not self.performance_data:
            return BottleneckAnalysis()

        bottlenecks = []
        recommendations = []

        # Analyze response times
        slow_operations = [
            m for m in self.performance_data if m.response_time_ms > 1000
        ]
        if slow_operations:
            bottlenecks.append("Slow response times detected")
            recommendations.append(
                "Optimize slow operations or implement caching"
            )

        # Analyze memory usage
        high_memory_operations = [
            m for m in self.performance_data if m.memory_usage_mb > 500
        ]
        if high_memory_operations:
            bottlenecks.append("High memory usage detected")
            recommendations.append("Implement memory optimization strategies")

        # Check performance requirements compliance
        page_load_violations = [
            m
            for m in self.performance_data
            if not m.meets_page_load_requirement
        ]
        config_violations = [
            m
            for m in self.performance_data
            if not m.meets_config_validation_requirement
        ]

        if page_load_violations:
            bottlenecks.append("Page load requirement violations (<2s)")
            recommendations.append("Optimize page loading mechanisms")

        if config_violations:
            bottlenecks.append(
                "Config validation requirement violations (<500ms)"
            )
            recommendations.append("Optimize configuration validation logic")

        return BottleneckAnalysis(
            identified_bottlenecks=bottlenecks,
            optimization_recommendations=recommendations,
            resource_constraints={
                "memory_limit_mb": 8192.0,
                "cpu_cores": float(psutil.cpu_count() or 1),
            },
            performance_regression_detected=len(bottlenecks) > 2,
        )

    def _compile_performance_metrics(self) -> dict[str, float]:
        """Compile aggregated performance metrics."""
        if not self.performance_data:
            return {}

        return {
            "total_measurements": float(len(self.performance_data)),
            "avg_response_time_ms": sum(
                m.response_time_ms for m in self.performance_data
            )
            / len(self.performance_data),
            "max_response_time_ms": max(
                m.response_time_ms for m in self.performance_data
            ),
            "avg_memory_usage_mb": sum(
                m.memory_usage_mb for m in self.performance_data
            )
            / len(self.performance_data),
            "peak_memory_usage_mb": max(
                m.memory_peak_mb for m in self.performance_data
            ),
            "avg_cpu_usage_percent": sum(
                m.cpu_usage_percent for m in self.performance_data
            )
            / len(self.performance_data),
            "page_load_compliance_rate": sum(
                1
                for m in self.performance_data
                if m.meets_page_load_requirement
            )
            / len(self.performance_data)
            * 100,
            "config_validation_compliance_rate": sum(
                1
                for m in self.performance_data
                if m.meets_config_validation_requirement
            )
            / len(self.performance_data)
            * 100,
            "overall_success_rate": sum(
                m.success_rate_percent for m in self.performance_data
            )
            / len(self.performance_data),
        }

    def _extract_error_details(self) -> list[str]:
        """Extract error details from performance measurements."""
        errors = []

        failed_measurements = [
            m for m in self.performance_data if m.success_rate_percent < 100.0
        ]
        for measurement in failed_measurements:
            errors.append(
                f"Performance failure in {measurement.test_scenario}"
            )

        requirement_failures = [
            m
            for m in self.performance_data
            if not m.meets_page_load_requirement
            or not m.meets_config_validation_requirement
        ]
        for failure in requirement_failures:
            errors.append(
                f"Performance requirement violation in {failure.test_scenario}"
            )

        return errors

    def _generate_performance_artifacts(
        self, config: AutomationConfiguration
    ) -> list[Path]:
        """Generate performance analysis artifacts."""
        artifacts = []

        # Performance metrics JSON export
        metrics_path = config.output_directory / "performance_metrics.json"
        # Will be implemented in the metrics export functionality
        artifacts.append(metrics_path)

        # Performance report HTML
        report_path = config.output_directory / "performance_report.html"
        artifacts.append(report_path)

        return artifacts

    def _validate_performance_requirements(self) -> dict[str, bool]:
        """Validate performance requirements compliance."""
        if not self.performance_data:
            return {"no_data": False}

        page_load_compliance = all(
            m.meets_page_load_requirement for m in self.performance_data
        )
        config_validation_compliance = all(
            m.meets_config_validation_requirement
            for m in self.performance_data
        )

        return {
            "page_load_requirements_met": page_load_compliance,
            "config_validation_requirements_met": config_validation_compliance,
            "all_requirements_met": page_load_compliance
            and config_validation_compliance,
        }

    def _simulate_memory_intensive_workflow(self) -> dict[str, Any]:
        """Simulate memory-intensive workflow for benchmarking."""
        # Simulate memory usage
        data = list(range(100000))  # Small memory allocation for testing
        return {"simulated_memory_operations": len(data)}

    def _simulate_cpu_intensive_workflow(self) -> dict[str, Any]:
        """Simulate CPU-intensive workflow for benchmarking."""
        # Simulate CPU usage
        result = sum(i * i for i in range(10000))  # CPU computation
        return {"simulated_cpu_operations": result}

    def _simulate_concurrent_users(self, user_count: int) -> dict[str, Any]:
        """Simulate concurrent user load for scalability testing."""
        # Simulate concurrent operations
        time.sleep(0.1 * user_count)  # Simulate increasing load
        return {"simulated_users": user_count, "load_simulation": "completed"}
