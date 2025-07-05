"""Test suite for performance benchmarking component.

This module provides comprehensive tests for the
PerformanceBenchmarkingComponent, validating performance measurement
capabilities, metrics collection, bottleneck
analysis, and integration with the automation framework.
"""

import time
from datetime import datetime

from ..test_base import WorkflowTestBase
from .performance_benchmarking import (
    BottleneckAnalysis,
    PerformanceBenchmarkingComponent,
    PerformanceInstrumentationMixin,
    PerformanceMetrics,
)


class TestPerformanceMetrics(WorkflowTestBase):
    """Test performance metrics data structures and validation."""

    def test_performance_metrics_creation(self) -> None:
        """Test PerformanceMetrics creation with valid data."""
        metrics = PerformanceMetrics(
            response_time_ms=150.5,
            page_load_time_ms=1200.0,
            config_validation_time_ms=250.0,
            workflow_execution_time_ms=800.0,
            cpu_usage_percent=45.2,
            memory_usage_mb=512.0,
            memory_peak_mb=768.0,
            disk_io_read_mb=10.5,
            disk_io_write_mb=5.2,
            operations_per_second=6.66,
            concurrent_users_supported=3,
            success_rate_percent=95.0,
            error_rate_percent=5.0,
            meets_page_load_requirement=True,
            meets_config_validation_requirement=True,
            test_scenario="test_config_load",
            workflow_phase="configuration",
        )

        assert metrics.response_time_ms == 150.5
        assert metrics.meets_page_load_requirement is True
        assert metrics.meets_config_validation_requirement is True
        assert metrics.test_scenario == "test_config_load"
        assert isinstance(metrics.timestamp, datetime)

    def test_performance_requirements_validation(self) -> None:
        """Test performance requirements validation logic."""
        # Test page load requirement compliance
        fast_metrics = PerformanceMetrics(
            response_time_ms=1500.0,  # 1.5s - should pass <2s requirement
            page_load_time_ms=1500.0,
            config_validation_time_ms=300.0,  # 0.3s - passes <500ms
            workflow_execution_time_ms=1500.0,
            cpu_usage_percent=20.0,
            memory_usage_mb=256.0,
            memory_peak_mb=300.0,
            disk_io_read_mb=1.0,
            disk_io_write_mb=0.5,
            operations_per_second=0.67,
            concurrent_users_supported=1,
            success_rate_percent=100.0,
            error_rate_percent=0.0,
            meets_page_load_requirement=True,
            meets_config_validation_requirement=True,
            test_scenario="fast_operation",
        )

        assert fast_metrics.meets_page_load_requirement is True
        assert fast_metrics.meets_config_validation_requirement is True

        # Test requirement violations
        slow_metrics = PerformanceMetrics(
            response_time_ms=3000.0,  # 3s - should fail <2s requirement
            page_load_time_ms=3000.0,
            config_validation_time_ms=800.0,  # 0.8s - fails <500ms
            workflow_execution_time_ms=3000.0,
            cpu_usage_percent=90.0,
            memory_usage_mb=1024.0,
            memory_peak_mb=1200.0,
            disk_io_read_mb=50.0,
            disk_io_write_mb=25.0,
            operations_per_second=0.33,
            concurrent_users_supported=1,
            success_rate_percent=80.0,
            error_rate_percent=20.0,
            meets_page_load_requirement=False,
            meets_config_validation_requirement=False,
            test_scenario="slow_operation",
        )

        assert slow_metrics.meets_page_load_requirement is False
        assert slow_metrics.meets_config_validation_requirement is False


class TestBottleneckAnalysis(WorkflowTestBase):
    """Test bottleneck analysis functionality."""

    def test_bottleneck_analysis_creation(self) -> None:
        """Test BottleneckAnalysis creation and data structure."""
        analysis = BottleneckAnalysis(
            identified_bottlenecks=["Slow database queries", "Memory leaks"],
            critical_path_components=["config_loader", "training_pipeline"],
            resource_constraints={"memory_limit_mb": 8192, "cpu_cores": 8},
            optimization_recommendations=[
                "Add database indexing",
                "Implement connection pooling",
            ],
            performance_regression_detected=True,
            baseline_comparison={
                "response_time_ms": 1.5,
                "memory_usage_mb": 2.1,
            },
        )

        assert len(analysis.identified_bottlenecks) == 2
        assert "Slow database queries" in analysis.identified_bottlenecks
        assert len(analysis.optimization_recommendations) == 2
        assert analysis.performance_regression_detected is True
        assert analysis.resource_constraints["memory_limit_mb"] == 8192

    def test_empty_bottleneck_analysis(self) -> None:
        """Test BottleneckAnalysis with default values."""
        analysis = BottleneckAnalysis()

        assert len(analysis.identified_bottlenecks) == 0
        assert len(analysis.optimization_recommendations) == 0
        assert analysis.performance_regression_detected is False
        assert len(analysis.resource_constraints) == 0


class TestPerformanceInstrumentationMixin(WorkflowTestBase):
    """Test performance instrumentation mixin functionality."""

    def setup_method(self) -> None:
        """Setup test environment with instrumentation mixin."""
        super().setup_method()
        self.instrumentation = PerformanceInstrumentationMixin()

    def test_measure_performance_success(self) -> None:
        """Test performance measurement of successful operations."""

        def test_operation(value: int) -> int:
            time.sleep(0.1)  # Simulate work
            return value * 2

        result, metrics = self.instrumentation.measure_performance(
            "test_operation", test_operation, 5
        )

        assert result == 10
        assert metrics.response_time_ms >= 100.0  # At least 100ms due to sleep
        assert metrics.success_rate_percent == 100.0
        assert metrics.error_rate_percent == 0.0
        assert metrics.test_scenario == "test_operation"
        assert len(self.instrumentation.performance_data) == 1

    def test_measure_performance_failure(self) -> None:
        """Test performance measurement of failed operations."""

        def failing_operation() -> None:
            time.sleep(0.05)
            raise ValueError("Test error")

        result, metrics = self.instrumentation.measure_performance(
            "failing_operation", failing_operation
        )

        assert isinstance(result, ValueError)
        assert metrics.response_time_ms >= 50.0  # At least 50ms due to sleep
        assert metrics.success_rate_percent == 0.0
        assert metrics.error_rate_percent == 100.0
        assert metrics.test_scenario == "failing_operation"

    def test_performance_requirements_detection(self) -> None:
        """Test detection of performance requirements compliance."""

        # Test config validation timing
        def fast_config_operation() -> dict[str, str]:
            time.sleep(0.2)  # 200ms - should pass <500ms requirement
            return {"status": "success"}

        result, metrics = self.instrumentation.measure_performance(
            "config_validation_test", fast_config_operation
        )

        assert metrics.meets_config_validation_requirement is True
        assert metrics.config_validation_time_ms >= 200.0

        # Test page load timing
        def slow_page_operation() -> dict[str, str]:
            time.sleep(2.5)  # 2.5s - should fail <2s requirement
            return {"page": "loaded"}

        result, metrics = self.instrumentation.measure_performance(
            "page_load_test", slow_page_operation
        )

        assert metrics.meets_page_load_requirement is False
        assert metrics.page_load_time_ms >= 2500.0


class TestPerformanceBenchmarkingComponent(WorkflowTestBase):
    """Test performance benchmarking component functionality."""

    def setup_method(self) -> None:
        """Setup test environment with performance benchmarking component."""
        super().setup_method()
        self.benchmarking_component = PerformanceBenchmarkingComponent(self)

    def test_component_initialization(self) -> None:
        """Test performance benchmarking component initialization."""
        assert self.benchmarking_component.test_utilities is self
        assert self.benchmarking_component.workflow_automation is not None
        assert hasattr(self.benchmarking_component, "performance_data")
        assert len(self.benchmarking_component.performance_data) == 0

    def test_workflow_name(self) -> None:
        """Test workflow name generation."""
        name = self.benchmarking_component.get_workflow_name()
        assert name == "CrackSeg Performance Benchmarking Suite"

    def test_preconditions_validation(self) -> None:
        """Test automation preconditions validation."""
        valid = self.benchmarking_component.validate_automation_preconditions()
        assert valid is True

    def test_automation_metrics_empty(self) -> None:
        """Test automation metrics with no performance data."""
        metrics = self.benchmarking_component.get_automation_metrics()
        assert "no_performance_data" in metrics
        assert metrics["no_performance_data"] == 0.0

    def test_automation_metrics_with_data(self) -> None:
        """Test automation metrics with performance data."""
        # Add some test performance data
        test_metrics = PerformanceMetrics(
            response_time_ms=200.0,
            page_load_time_ms=1500.0,
            config_validation_time_ms=300.0,
            workflow_execution_time_ms=800.0,
            cpu_usage_percent=30.0,
            memory_usage_mb=400.0,
            memory_peak_mb=450.0,
            disk_io_read_mb=5.0,
            disk_io_write_mb=2.0,
            operations_per_second=5.0,
            concurrent_users_supported=2,
            success_rate_percent=95.0,
            error_rate_percent=5.0,
            meets_page_load_requirement=True,
            meets_config_validation_requirement=True,
            test_scenario="test_scenario",
        )

        self.benchmarking_component.performance_data.append(test_metrics)

        metrics = self.benchmarking_component.get_automation_metrics()

        assert "avg_response_time_ms" in metrics
        assert metrics["avg_response_time_ms"] == 200.0
        assert "avg_memory_usage_mb" in metrics
        assert metrics["avg_memory_usage_mb"] == 400.0
        assert "performance_measurements" in metrics
        assert metrics["performance_measurements"] == 1.0
        assert "page_load_compliance_rate" in metrics
        assert metrics["page_load_compliance_rate"] == 100.0

    def test_bottleneck_analysis_no_data(self) -> None:
        """Test bottleneck analysis with no performance data."""
        analysis = (
            self.benchmarking_component._analyze_performance_bottlenecks()
        )

        assert isinstance(analysis, BottleneckAnalysis)
        assert len(analysis.identified_bottlenecks) == 0
        assert len(analysis.optimization_recommendations) == 0
        assert analysis.performance_regression_detected is False

    def test_bottleneck_analysis_with_issues(self) -> None:
        """Test bottleneck analysis with performance issues."""
        # Add problematic performance data
        slow_metrics = PerformanceMetrics(
            response_time_ms=2500.0,  # Slow response
            page_load_time_ms=3000.0,  # Page load violation
            config_validation_time_ms=800.0,  # Config validation violation
            workflow_execution_time_ms=2500.0,
            cpu_usage_percent=90.0,
            memory_usage_mb=600.0,  # High memory usage
            memory_peak_mb=700.0,
            disk_io_read_mb=100.0,
            disk_io_write_mb=50.0,
            operations_per_second=0.4,
            concurrent_users_supported=1,
            success_rate_percent=70.0,
            error_rate_percent=30.0,
            meets_page_load_requirement=False,
            meets_config_validation_requirement=False,
            test_scenario="problematic_scenario",
        )

        self.benchmarking_component.performance_data.append(slow_metrics)

        analysis = (
            self.benchmarking_component._analyze_performance_bottlenecks()
        )

        assert len(analysis.identified_bottlenecks) > 0
        assert len(analysis.optimization_recommendations) > 0
        assert analysis.performance_regression_detected is True
        assert (
            "Slow response times detected" in analysis.identified_bottlenecks
        )
        assert (
            "Page load requirement violations (<2s)"
            in analysis.identified_bottlenecks
        )
        assert (
            "Config validation requirement violations (<500ms)"
            in analysis.identified_bottlenecks
        )

    def test_performance_metrics_compilation(self) -> None:
        """Test compilation of performance metrics."""
        # Add multiple performance measurements
        for i in range(3):
            metrics = PerformanceMetrics(
                response_time_ms=100.0 + i * 50,
                page_load_time_ms=1000.0 + i * 200,
                config_validation_time_ms=200.0 + i * 50,
                workflow_execution_time_ms=500.0 + i * 100,
                cpu_usage_percent=20.0 + i * 10,
                memory_usage_mb=300.0 + i * 50,
                memory_peak_mb=350.0 + i * 50,
                disk_io_read_mb=2.0 + i,
                disk_io_write_mb=1.0 + i * 0.5,
                operations_per_second=5.0 - i * 0.5,
                concurrent_users_supported=1 + i,
                success_rate_percent=100.0 - i * 5,
                error_rate_percent=i * 5,
                meets_page_load_requirement=True,
                meets_config_validation_requirement=True,
                test_scenario=f"test_scenario_{i}",
            )
            self.benchmarking_component.performance_data.append(metrics)

        compiled_metrics = (
            self.benchmarking_component._compile_performance_metrics()
        )

        assert compiled_metrics["total_measurements"] == 3.0
        assert (
            compiled_metrics["avg_response_time_ms"] == 150.0
        )  # (100+150+200)/3
        assert compiled_metrics["max_response_time_ms"] == 200.0
        assert (
            compiled_metrics["avg_memory_usage_mb"] == 350.0
        )  # (300+350+400)/3
        assert compiled_metrics["page_load_compliance_rate"] == 100.0
        assert compiled_metrics["config_validation_compliance_rate"] == 100.0

    def test_performance_requirements_validation(self) -> None:
        """Test performance requirements validation."""
        # Add mixed compliance data
        compliant_metrics = PerformanceMetrics(
            response_time_ms=1500.0,
            page_load_time_ms=1500.0,
            config_validation_time_ms=300.0,
            workflow_execution_time_ms=1500.0,
            cpu_usage_percent=25.0,
            memory_usage_mb=300.0,
            memory_peak_mb=350.0,
            disk_io_read_mb=2.0,
            disk_io_write_mb=1.0,
            operations_per_second=0.67,
            concurrent_users_supported=1,
            success_rate_percent=100.0,
            error_rate_percent=0.0,
            meets_page_load_requirement=True,
            meets_config_validation_requirement=True,
            test_scenario="compliant_test",
        )

        non_compliant_metrics = PerformanceMetrics(
            response_time_ms=3000.0,
            page_load_time_ms=3000.0,
            config_validation_time_ms=800.0,
            workflow_execution_time_ms=3000.0,
            cpu_usage_percent=80.0,
            memory_usage_mb=800.0,
            memory_peak_mb=900.0,
            disk_io_read_mb=50.0,
            disk_io_write_mb=25.0,
            operations_per_second=0.33,
            concurrent_users_supported=1,
            success_rate_percent=70.0,
            error_rate_percent=30.0,
            meets_page_load_requirement=False,
            meets_config_validation_requirement=False,
            test_scenario="non_compliant_test",
        )

        self.benchmarking_component.performance_data.extend(
            [compliant_metrics, non_compliant_metrics]
        )

        validation_results = (
            self.benchmarking_component._validate_performance_requirements()
        )

        assert validation_results["page_load_requirements_met"] is False
        assert (
            validation_results["config_validation_requirements_met"] is False
        )
        assert validation_results["all_requirements_met"] is False
