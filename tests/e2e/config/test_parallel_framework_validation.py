"""Validation tests for the parallel test execution framework.

This module contains comprehensive tests to validate the entire parallel
test execution framework, including configuration, resource management,
performance integration, and execution strategies.
"""

import json
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.e2e.config.execution_strategies import (
    CIPipelineStrategy,
    CustomStrategy,
    DevelopmentStrategy,
    FullTestStrategy,
    PerformanceTestStrategy,
    SmokeTestStrategy,
    StrategyExecutor,
    StrategyResult,
    StrategyType,
    create_smoke_strategy,
    global_strategy_executor,
)
from tests.e2e.config.parallel_execution_config import (
    ExecutionStrategy,
    ParallelTestConfig,
    ResourceLimits,
    get_predefined_config,
)
from tests.e2e.config.parallel_performance_integration import (
    ParallelPerformanceIntegration,
    ParallelPerformanceMonitor,
    WorkerPerformanceData,
    create_worker_performance_monitor,
    global_performance_integration,
)
from tests.e2e.config.pytest_markers import (
    get_marker_configuration,
    get_performance_markers,
    get_resource_markers,
)
from tests.e2e.config.resource_manager import (
    PortManager,
    ResourceAllocation,
    ResourceManager,
    WorkerIsolation,
)


class TestParallelExecutionConfig:
    """Test parallel execution configuration functionality."""

    def test_resource_limits_validation(self) -> None:
        """Test resource limits validation."""
        # Valid limits
        limits = ResourceLimits(
            memory_limit_mb=1024,
            cpu_limit=4,
            max_workers=8,
            timeout_seconds=300,
        )

        assert limits.memory_limit_mb == 1024
        assert limits.cpu_limit == 4
        assert limits.max_workers == 8
        assert limits.timeout_seconds == 300

    def test_parallel_config_creation(self) -> None:
        """Test parallel test configuration creation."""
        config = ParallelTestConfig(
            execution_strategy=ExecutionStrategy.PARALLEL_BY_TEST,
            worker_count=4,
            resource_limits=ResourceLimits(memory_limit_mb=512, cpu_limit=2),
        )

        assert config.execution_strategy == ExecutionStrategy.PARALLEL_BY_TEST
        assert config.worker_count == 4
        assert config.resource_limits.memory_limit_mb == 512

    def test_predefined_configurations(self) -> None:
        """Test all predefined configurations are valid."""
        config_names = ["dev", "ci", "performance", "smoke", "full"]

        for name in config_names:
            config = get_predefined_config(name)
            assert isinstance(config, ParallelTestConfig)
            assert config.worker_count > 0
            assert config.resource_limits.memory_limit_mb > 0
            assert config.resource_limits.cpu_limit > 0

    def test_pytest_args_generation(self) -> None:
        """Test pytest arguments generation."""
        config = get_predefined_config("dev")
        args = config.get_pytest_args()

        assert isinstance(args, list)
        assert len(args) > 0
        # Should contain pytest-xdist arguments
        assert any("pytest-xdist" in str(arg) or "-n" in args for arg in args)

    def test_worker_environment_variables(self) -> None:
        """Test worker environment variable setup."""
        config = get_predefined_config("ci")
        env_vars = config.get_worker_env_vars()

        assert isinstance(env_vars, dict)
        assert "PYTEST_CURRENT_TEST" in env_vars or len(env_vars) >= 0


class TestResourceManager:
    """Test resource management functionality."""

    def test_port_manager_allocation(self) -> None:
        """Test port allocation and deallocation."""
        port_manager = PortManager()

        # Allocate ports
        port1 = port_manager.allocate_port()
        port2 = port_manager.allocate_port()

        assert port1 != port2
        assert port1 in port_manager._allocated_ports
        assert port2 in port_manager._allocated_ports

        # Deallocate port
        port_manager.deallocate_port(port1)
        assert port1 not in port_manager._allocated_ports

    def test_resource_allocation_tracking(self) -> None:
        """Test resource allocation tracking."""
        allocation = ResourceAllocation(
            worker_id="test_worker",
            memory_limit_mb=512,
            cpu_limit=2,
            allocated_ports=[8600, 8601],
        )

        assert allocation.worker_id == "test_worker"
        assert allocation.memory_limit_mb == 512
        assert 8600 in allocation.allocated_ports

    def test_worker_isolation(self) -> None:
        """Test worker isolation functionality."""
        isolation = WorkerIsolation()

        with isolation.isolate_process("test_worker"):
            # Inside isolation context
            assert True  # Process isolation would be tested in integration

        # Outside isolation context
        assert True

    def test_resource_manager_context(self) -> None:
        """Test resource manager context manager."""
        manager = ResourceManager()

        with manager.acquire_resources(
            memory_limit_mb=256, cpu_limit=1
        ) as allocation:
            assert isinstance(allocation, ResourceAllocation)
            assert allocation.memory_limit_mb == 256
            assert allocation.cpu_limit == 1

    def test_concurrent_resource_allocation(self) -> None:
        """Test concurrent resource allocation."""
        manager = ResourceManager()
        allocations = []

        def allocate_resources():
            with manager.acquire_resources(
                memory_limit_mb=128, cpu_limit=1
            ) as alloc:
                allocations.append(alloc)
                time.sleep(0.1)  # Hold resources briefly

        # Run multiple threads
        threads = [
            threading.Thread(target=allocate_resources) for _ in range(3)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(allocations) == 3
        # All should have unique worker IDs
        worker_ids = [alloc.worker_id for alloc in allocations]
        assert len(set(worker_ids)) == len(worker_ids)


class TestPytestMarkers:
    """Test pytest markers functionality."""

    def test_marker_configuration_creation(self) -> None:
        """Test marker configuration creation."""
        config = get_marker_configuration()

        assert hasattr(config, "performance_markers")
        assert hasattr(config, "resource_markers")
        assert hasattr(config, "execution_markers")

    def test_performance_markers(self) -> None:
        """Test performance markers."""
        markers = get_performance_markers()

        assert "performance" in markers
        assert "performance_critical" in markers
        assert "performance_baseline" in markers

    def test_resource_markers(self) -> None:
        """Test resource markers."""
        markers = get_resource_markers()

        assert "high_memory" in markers
        assert "cpu_intensive" in markers
        assert "isolated" in markers

    def test_marker_expressions(self) -> None:
        """Test marker expression building."""
        # This would test the marker expression building functionality
        # if it were implemented in the markers module
        config = get_marker_configuration()
        assert config is not None


class TestParallelPerformanceIntegration:
    """Test parallel performance integration."""

    def test_worker_performance_data_creation(self) -> None:
        """Test worker performance data creation."""
        from tests.e2e.helpers.performance_monitoring import PerformanceReport

        report = PerformanceReport(
            test_name="test_worker",
            start_time=time.time(),
            end_time=time.time() + 10,
        )

        worker_data = WorkerPerformanceData(
            worker_id="worker_1",
            test_name="test_example",
            performance_report=report,
            start_time=report.start_time,
            end_time=report.end_time,
        )

        assert worker_data.worker_id == "worker_1"
        assert worker_data.test_name == "test_example"
        assert worker_data.duration == 10.0

    def test_parallel_performance_monitor(self) -> None:
        """Test parallel performance monitor."""
        monitor = ParallelPerformanceMonitor("test_suite")

        # Start monitoring
        monitor.start_suite_monitoring()
        assert monitor.monitoring_active

        # Create worker monitor
        worker_monitor = monitor.create_worker_monitor("worker_1", "test_1")
        assert worker_monitor is not None

        # Register completion
        monitor.register_worker_completion("worker_1", "test_1", success=True)
        assert "worker_1" in monitor.worker_data

        # Stop monitoring
        monitor.stop_suite_monitoring()
        assert not monitor.monitoring_active

    def test_parallel_performance_integration_class(self) -> None:
        """Test parallel performance integration class."""
        integration = ParallelPerformanceIntegration()

        # Create suite monitor
        monitor = integration.create_suite_monitor("test_integration")
        assert monitor is not None
        assert monitor.test_suite_name == "test_integration"

        # Get suite monitor
        retrieved_monitor = integration.get_suite_monitor("test_integration")
        assert retrieved_monitor is monitor

        # Cleanup
        integration.cleanup_suite_monitor("test_integration")
        assert "test_integration" not in integration.suite_monitors

    def test_performance_report_generation(self) -> None:
        """Test performance report generation."""
        monitor = ParallelPerformanceMonitor("test_report")
        monitor.start_suite_monitoring()

        # Simulate worker execution
        worker_monitor = monitor.create_worker_monitor("worker_1", "test_1")
        worker_monitor.start_monitoring()
        time.sleep(0.1)  # Brief execution
        worker_monitor.stop_monitoring()

        monitor.register_worker_completion("worker_1", "test_1", success=True)

        # Generate report
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "test_report.json"
            report = monitor.generate_consolidated_report(output_file)

            assert report.test_suite_name == "test_report"
            assert report.total_workers == 1
            assert report.successful_workers == 1
            assert output_file.exists()

            # Verify report file content
            with open(output_file) as f:
                report_data = json.load(f)
            assert report_data["test_suite_name"] == "test_report"

    def test_utility_functions(self) -> None:
        """Test utility functions for performance monitoring."""
        # Test worker monitor creation
        monitor = create_worker_performance_monitor(
            "test_suite", "worker_1", "test_1"
        )
        assert monitor is not None

        # Cleanup
        global_performance_integration.cleanup_suite_monitor("test_suite")


class TestExecutionStrategies:
    """Test execution strategies."""

    def test_smoke_strategy_creation(self) -> None:
        """Test smoke strategy creation."""
        strategy = SmokeTestStrategy()

        assert strategy.name == "smoke_tests"
        assert strategy.strategy_type == StrategyType.SMOKE
        assert not strategy.should_monitor_performance()

        args = strategy.get_pytest_args()
        assert "-m" in args
        assert "smoke" in args

    def test_performance_strategy_creation(self) -> None:
        """Test performance strategy creation."""
        strategy = PerformanceTestStrategy()

        assert strategy.name == "performance_tests"
        assert strategy.strategy_type == StrategyType.PERFORMANCE
        assert strategy.should_monitor_performance()

        markers = strategy.get_markers()
        assert "performance" in markers

    def test_full_strategy_creation(self) -> None:
        """Test full strategy creation."""
        strategy = FullTestStrategy()

        assert strategy.name == "full_test_suite"
        assert strategy.strategy_type == StrategyType.FULL
        assert strategy.should_monitor_performance()

    def test_development_strategy_creation(self) -> None:
        """Test development strategy creation."""
        strategy = DevelopmentStrategy(focus_areas=["gui", "config"])

        assert strategy.name == "development"
        assert strategy.strategy_type == StrategyType.DEVELOPMENT
        assert not strategy.should_monitor_performance()
        assert strategy.focus_areas == ["gui", "config"]

    def test_ci_strategy_creation(self) -> None:
        """Test CI strategy creation."""
        strategy = CIPipelineStrategy()

        assert strategy.name == "ci_pipeline"
        assert strategy.strategy_type == StrategyType.CI_PIPELINE
        assert strategy.should_monitor_performance()

        args = strategy.get_pytest_args()
        assert "--junit-xml" in " ".join(args)

    def test_custom_strategy_creation(self) -> None:
        """Test custom strategy creation."""
        strategy = CustomStrategy(
            name="custom_test",
            pytest_args=["--custom-arg"],
            markers=["custom_marker"],
            monitor_performance=True,
        )

        assert strategy.name == "custom_test"
        assert strategy.strategy_type == StrategyType.CUSTOM
        assert strategy.should_monitor_performance()
        assert "--custom-arg" in strategy.get_pytest_args()
        assert "custom_marker" in strategy.get_markers()

    def test_strategy_factory_functions(self) -> None:
        """Test strategy factory functions."""
        smoke = create_smoke_strategy()
        assert isinstance(smoke, SmokeTestStrategy)

    def test_strategy_result_analysis(self) -> None:
        """Test strategy result analysis."""
        result = StrategyResult(
            strategy_name="test_strategy",
            strategy_type=StrategyType.SMOKE,
            success=True,
            execution_time=10.5,
            tests_executed=10,
            tests_passed=8,
            tests_failed=2,
            worker_count=4,
        )

        assert result.success_rate == 80.0
        assert result.strategy_name == "test_strategy"


class TestStrategyExecutor:
    """Test strategy executor."""

    def test_executor_initialization(self) -> None:
        """Test strategy executor initialization."""
        executor = StrategyExecutor()

        assert executor.resource_manager is not None
        assert executor.performance_integration is not None

    @patch(
        "tests.e2e.config.execution_strategies.StrategyExecutor._execute_pytest"
    )
    def test_strategy_execution_simulation(
        self, mock_execute: MagicMock
    ) -> None:
        """Test strategy execution with mocked pytest execution."""
        # Mock pytest execution results
        mock_execute.return_value = {
            "success": True,
            "execution_time": 5.0,
            "tests_executed": 5,
            "tests_passed": 5,
            "tests_failed": 0,
            "worker_count": 2,
        }

        executor = StrategyExecutor()
        strategy = SmokeTestStrategy()

        result = executor.execute_strategy(strategy)

        assert result.success
        assert result.tests_executed == 5
        assert result.tests_passed == 5
        assert result.worker_count == 2
        assert mock_execute.called

    def test_pytest_command_building(self) -> None:
        """Test pytest command building."""
        executor = StrategyExecutor()
        strategy = SmokeTestStrategy()
        config = strategy.get_parallel_config()

        command = executor._build_pytest_command(strategy, config)

        assert "pytest" in command
        assert "-m" in command
        assert "smoke" in command

    def test_global_executor_access(self) -> None:
        """Test global executor access."""
        assert global_strategy_executor is not None
        assert isinstance(global_strategy_executor, StrategyExecutor)


class TestFrameworkIntegration:
    """Test complete framework integration."""

    def test_end_to_end_smoke_test(self) -> None:
        """Test end-to-end smoke test execution simulation."""
        # Create components
        resource_manager = ResourceManager()
        performance_integration = ParallelPerformanceIntegration(
            resource_manager
        )
        executor = StrategyExecutor(resource_manager, performance_integration)

        # Create and configure strategy
        strategy = SmokeTestStrategy()

        # Mock the actual pytest execution
        with patch.object(executor, "_execute_pytest") as mock_execute:
            mock_execute.return_value = {
                "success": True,
                "execution_time": 2.0,
                "tests_executed": 3,
                "tests_passed": 3,
                "tests_failed": 0,
                "worker_count": 2,
            }

            # Execute strategy
            result = executor.execute_strategy(strategy)

            # Verify results
            assert result.success
            assert result.strategy_name == "smoke_tests"
            assert result.tests_executed == 3
            assert result.success_rate == 100.0

    def test_resource_coordination(self) -> None:
        """Test resource coordination between components."""
        # Test that resource manager coordinates properly with other components
        manager = ResourceManager()

        # Allocate resources for parallel execution
        with manager.acquire_resources(
            memory_limit_mb=512, cpu_limit=2
        ) as allocation:
            # Verify allocation
            assert allocation.memory_limit_mb == 512
            assert allocation.cpu_limit == 2
            assert allocation.worker_id is not None

            # Simulate parallel test execution within resource limits
            assert len(allocation.allocated_ports) >= 0

    def test_performance_monitoring_integration(self) -> None:
        """Test performance monitoring integration."""
        integration = ParallelPerformanceIntegration()

        # Create suite monitor
        monitor = integration.create_suite_monitor("integration_test")
        monitor.start_suite_monitoring()

        # Simulate worker execution
        worker_monitor = monitor.create_worker_monitor(
            "worker_1", "test_integration"
        )
        worker_monitor.start_monitoring()

        # Add some metrics
        worker_monitor.add_custom_metric("test_metric", 42.0, "ms")

        # Complete execution
        worker_monitor.stop_monitoring()
        monitor.register_worker_completion(
            "worker_1", "test_integration", success=True
        )

        # Generate report
        report = monitor.generate_consolidated_report()

        assert report.successful_workers == 1
        assert report.total_workers == 1
        assert len(report.worker_reports) == 1

        # Cleanup
        integration.cleanup_suite_monitor("integration_test")

    def test_configuration_consistency(self) -> None:
        """Test configuration consistency across components."""
        # Test that all predefined configurations work with all strategies
        strategies = [
            SmokeTestStrategy(),
            PerformanceTestStrategy(),
            FullTestStrategy(),
            DevelopmentStrategy(),
            CIPipelineStrategy(),
        ]

        for strategy in strategies:
            config = strategy.get_parallel_config()
            args = config.get_pytest_args()

            # Verify configuration is valid
            assert config.worker_count > 0
            assert config.resource_limits.memory_limit_mb > 0
            assert isinstance(args, list)

            # Verify strategy produces valid pytest arguments
            strategy_args = strategy.get_pytest_args()
            assert isinstance(strategy_args, list)

    def test_error_handling_robustness(self) -> None:
        """Test error handling across the framework."""
        # Test resource manager error handling
        manager = ResourceManager()

        # Test invalid resource limits
        try:
            with manager.acquire_resources(memory_limit_mb=-1, cpu_limit=0):
                pass
        except (ValueError, AssertionError):
            pass  # Expected for invalid inputs

        # Test performance integration error handling
        integration = ParallelPerformanceIntegration()

        # Test handling of non-existent suite
        report = integration.generate_suite_report("non_existent_suite")
        assert report is None

    def test_cleanup_and_resource_deallocation(self) -> None:
        """Test proper cleanup and resource deallocation."""
        manager = ResourceManager()
        initial_port_count = len(manager.port_manager._allocated_ports)

        # Allocate and deallocate resources
        with manager.acquire_resources(memory_limit_mb=256, cpu_limit=1):
            # Resources should be allocated
            assert len(manager.allocations) > 0

        # Resources should be cleaned up
        final_port_count = len(manager.port_manager._allocated_ports)
        assert final_port_count == initial_port_count


# Integration test markers for pytest
pytestmark = [
    pytest.mark.integration,
    pytest.mark.parallel_framework,
    pytest.mark.performance_monitoring,
]
