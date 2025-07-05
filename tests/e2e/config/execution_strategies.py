"""Execution strategies for parallel test framework.

This module defines specific execution strategies that combine configuration,
markers, performance monitoring, and resource management for different
testing scenarios like smoke tests, performance tests, and full test suites.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from tests.e2e.config.parallel_execution_config import (
    ParallelTestConfig,
    get_predefined_config,
)
from tests.e2e.config.parallel_performance_integration import (
    global_performance_integration,
)
from tests.e2e.config.resource_manager import (
    ResourceManager,
    global_resource_manager,
)

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Types of execution strategies."""

    SMOKE = "smoke"
    PERFORMANCE = "performance"
    FULL = "full"
    CUSTOM = "custom"
    CI_PIPELINE = "ci_pipeline"
    DEVELOPMENT = "development"


@dataclass
class StrategyResult:
    """Results from executing a test strategy."""

    strategy_name: str
    strategy_type: StrategyType
    success: bool
    execution_time: float
    tests_executed: int
    tests_passed: int
    tests_failed: int
    worker_count: int
    performance_summary: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None
    detailed_results: dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.tests_executed == 0:
            return 0.0
        return (self.tests_passed / self.tests_executed) * 100


class BaseExecutionStrategy(ABC):
    """Abstract base class for test execution strategies."""

    def __init__(self, name: str, strategy_type: StrategyType) -> None:
        """Initialize execution strategy.

        Args:
            name: Name of the strategy
            strategy_type: Type of strategy
        """
        self.name = name
        self.strategy_type = strategy_type
        self.logger = logging.getLogger(f"{__name__}.{name}")

    @abstractmethod
    def get_pytest_args(self) -> list[str]:
        """Get pytest arguments for this strategy.

        Returns:
            List of pytest command-line arguments
        """

    @abstractmethod
    def get_parallel_config(self) -> ParallelTestConfig:
        """Get parallel test configuration for this strategy.

        Returns:
            ParallelTestConfig instance
        """

    @abstractmethod
    def should_monitor_performance(self) -> bool:
        """Determine if performance monitoring should be enabled.

        Returns:
            True if performance monitoring should be enabled
        """

    def get_markers(self) -> list[str]:
        """Get pytest markers for this strategy.

        Returns:
            List of marker expressions
        """
        return []

    def get_test_patterns(self) -> list[str]:
        """Get test file patterns for this strategy.

        Returns:
            List of test file patterns
        """
        return ["tests/e2e/tests/"]

    def pre_execution_setup(self) -> None:
        """Perform any setup before test execution."""
        self.logger.info(f"Setting up {self.name} strategy")

    def post_execution_cleanup(self, result: StrategyResult) -> None:
        """Perform cleanup after test execution.

        Args:
            result: Execution result for analysis
        """
        self.logger.info(
            f"Cleaning up {self.name} strategy - "
            f"Success: {result.success}, "
            f"Tests: {result.tests_executed}, "
            f"Duration: {result.execution_time:.2f}s"
        )

    def analyze_results(self, raw_results: dict[str, Any]) -> StrategyResult:
        """Analyze execution results and create strategy result.

        Args:
            raw_results: Raw execution results from test runner

        Returns:
            StrategyResult with analyzed data
        """
        return StrategyResult(
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            success=raw_results.get("success", False),
            execution_time=raw_results.get("execution_time", 0.0),
            tests_executed=raw_results.get("tests_executed", 0),
            tests_passed=raw_results.get("tests_passed", 0),
            tests_failed=raw_results.get("tests_failed", 0),
            worker_count=raw_results.get("worker_count", 1),
            performance_summary=raw_results.get("performance_summary", {}),
            error_message=raw_results.get("error_message"),
            detailed_results=raw_results,
        )


class SmokeTestStrategy(BaseExecutionStrategy):
    """Strategy for fast smoke tests to verify basic functionality."""

    def __init__(self) -> None:
        """Initialize smoke test strategy."""
        super().__init__("smoke_tests", StrategyType.SMOKE)

    def get_pytest_args(self) -> list[str]:
        """Get pytest arguments for smoke tests."""
        return [
            "-m",
            "smoke",
            "--tb=short",
            "--durations=10",
            "--maxfail=3",
            "-v",
        ]

    def get_parallel_config(self) -> ParallelTestConfig:
        """Get parallel configuration optimized for smoke tests."""
        return get_predefined_config("smoke")

    def should_monitor_performance(self) -> bool:
        """Smoke tests use minimal performance monitoring."""
        return False

    def get_markers(self) -> list[str]:
        """Get markers for smoke tests."""
        return ["smoke", "not slow", "not performance_intensive"]

    def get_test_patterns(self) -> list[str]:
        """Get test patterns for smoke tests."""
        return [
            "tests/e2e/tests/test_basic_*.py",
            "tests/e2e/tests/test_smoke_*.py",
        ]


class PerformanceTestStrategy(BaseExecutionStrategy):
    """Strategy for comprehensive performance testing."""

    def __init__(self) -> None:
        """Initialize performance test strategy."""
        super().__init__("performance_tests", StrategyType.PERFORMANCE)

    def get_pytest_args(self) -> list[str]:
        """Get pytest arguments for performance tests."""
        return [
            "-m",
            "performance",
            "--tb=long",
            "--durations=0",
            "--capture=no",
            "-v",
            "-s",
        ]

    def get_parallel_config(self) -> ParallelTestConfig:
        """Get parallel configuration optimized for performance tests."""
        return get_predefined_config("performance")

    def should_monitor_performance(self) -> bool:
        """Performance tests require detailed monitoring."""
        return True

    def get_markers(self) -> list[str]:
        """Get markers for performance tests."""
        return ["performance", "performance_critical", "not quick"]

    def get_test_patterns(self) -> list[str]:
        """Get test patterns for performance tests."""
        return [
            "tests/e2e/tests/test_performance_*.py",
            "tests/e2e/tests/test_load_*.py",
        ]

    def pre_execution_setup(self) -> None:
        """Setup for performance testing."""
        super().pre_execution_setup()
        # Ensure clean state for accurate performance measurements
        import gc

        gc.collect()
        self.logger.info("Performance test environment prepared")


class FullTestStrategy(BaseExecutionStrategy):
    """Strategy for executing the complete test suite."""

    def __init__(self) -> None:
        """Initialize full test strategy."""
        super().__init__("full_test_suite", StrategyType.FULL)

    def get_pytest_args(self) -> list[str]:
        """Get pytest arguments for full test suite."""
        return [
            "--tb=short",
            "--durations=20",
            "--cov=tests/e2e",
            "--cov-report=html",
            "--cov-report=term-missing",
            "-v",
        ]

    def get_parallel_config(self) -> ParallelTestConfig:
        """Get parallel configuration for full test suite."""
        return get_predefined_config("full")

    def should_monitor_performance(self) -> bool:
        """Full suite includes performance monitoring."""
        return True

    def get_markers(self) -> list[str]:
        """Get markers for full test suite (no exclusions)."""
        return []

    def get_test_patterns(self) -> list[str]:
        """Get test patterns for full suite."""
        return ["tests/e2e/tests/"]


class DevelopmentStrategy(BaseExecutionStrategy):
    """Strategy optimized for development workflow."""

    def __init__(self, focus_areas: list[str] | None = None) -> None:
        """Initialize development strategy.

        Args:
            focus_areas: Optional list of areas to focus on
                (e.g., ["gui", "config"])
        """
        super().__init__("development", StrategyType.DEVELOPMENT)
        self.focus_areas = focus_areas or []

    def get_pytest_args(self) -> list[str]:
        """Get pytest arguments for development testing."""
        args = [
            "--tb=short",
            "--durations=5",
            "--maxfail=1",
            "-v",
            "-x",  # Stop on first failure for quick feedback
        ]

        # Add focus area markers if specified
        if self.focus_areas:
            focus_marker = " or ".join(self.focus_areas)
            args.extend(["-m", focus_marker])

        return args

    def get_parallel_config(self) -> ParallelTestConfig:
        """Get parallel configuration for development."""
        return get_predefined_config("dev")

    def should_monitor_performance(self) -> bool:
        """Development tests use minimal monitoring for speed."""
        return False

    def get_markers(self) -> list[str]:
        """Get markers for development tests."""
        markers = ["not slow", "not performance_intensive"]
        if self.focus_areas:
            markers.extend(self.focus_areas)
        return markers


class CIPipelineStrategy(BaseExecutionStrategy):
    """Strategy optimized for CI/CD pipeline execution."""

    def __init__(self) -> None:
        """Initialize CI pipeline strategy."""
        super().__init__("ci_pipeline", StrategyType.CI_PIPELINE)

    def get_pytest_args(self) -> list[str]:
        """Get pytest arguments for CI pipeline."""
        return [
            "--tb=short",
            "--durations=20",
            "--maxfail=5",
            "--cov=tests/e2e",
            "--cov-report=xml",
            "--cov-report=term",
            "--junit-xml=test-results/junit.xml",
            "-v",
        ]

    def get_parallel_config(self) -> ParallelTestConfig:
        """Get parallel configuration for CI."""
        return get_predefined_config("ci")

    def should_monitor_performance(self) -> bool:
        """CI includes performance monitoring for analysis."""
        return True

    def get_markers(self) -> list[str]:
        """Get markers for CI tests."""
        return ["not manual", "not local_only"]


class CustomStrategy(BaseExecutionStrategy):
    """Fully customizable strategy for specific needs."""

    def __init__(
        self,
        name: str,
        pytest_args: list[str] | None = None,
        markers: list[str] | None = None,
        test_patterns: list[str] | None = None,
        parallel_config: ParallelTestConfig | None = None,
        monitor_performance: bool = False,
    ) -> None:
        """Initialize custom strategy.

        Args:
            name: Name of the custom strategy
            pytest_args: Custom pytest arguments
            markers: Custom pytest markers
            test_patterns: Custom test file patterns
            parallel_config: Custom parallel configuration
            monitor_performance: Whether to monitor performance
        """
        super().__init__(name, StrategyType.CUSTOM)
        self._pytest_args = pytest_args or []
        self._markers = markers or []
        self._test_patterns = test_patterns or ["tests/e2e/tests/"]
        self._parallel_config = parallel_config or get_predefined_config("dev")
        self._monitor_performance = monitor_performance

    def get_pytest_args(self) -> list[str]:
        """Get custom pytest arguments."""
        return self._pytest_args

    def get_parallel_config(self) -> ParallelTestConfig:
        """Get custom parallel configuration."""
        return self._parallel_config

    def should_monitor_performance(self) -> bool:
        """Return custom performance monitoring setting."""
        return self._monitor_performance

    def get_markers(self) -> list[str]:
        """Get custom markers."""
        return self._markers

    def get_test_patterns(self) -> list[str]:
        """Get custom test patterns."""
        return self._test_patterns


class StrategyExecutor:
    """Executor for running test strategies with full integration."""

    def __init__(
        self,
        resource_manager: ResourceManager | None = None,
        performance_integration: Any | None = None,
    ) -> None:
        """Initialize strategy executor.

        Args:
            resource_manager: Optional resource manager
            performance_integration: Optional performance integration
        """
        self.resource_manager = resource_manager or global_resource_manager
        self.performance_integration = (
            performance_integration or global_performance_integration
        )
        self.logger = logging.getLogger(__name__)

    def execute_strategy(
        self,
        strategy: BaseExecutionStrategy,
        output_dir: Path | str | None = None,
    ) -> StrategyResult:
        """Execute a test strategy with full framework integration.

        Args:
            strategy: Strategy to execute
            output_dir: Optional directory for output files

        Returns:
            StrategyResult with execution results
        """
        self.logger.info(f"Executing strategy: {strategy.name}")

        # Initialize components
        performance_monitor = None
        if strategy.should_monitor_performance():
            performance_monitor = (
                self.performance_integration.create_suite_monitor(
                    strategy.name
                )
            )
            performance_monitor.start_suite_monitoring()

        # Setup resource management
        parallel_config = strategy.get_parallel_config()
        with self.resource_manager.acquire_resources(
            memory_limit_mb=parallel_config.resource_limits.max_memory_mb,
            cpu_limit=int(parallel_config.resource_limits.max_cpu_percent),
        ):
            try:
                # Pre-execution setup
                strategy.pre_execution_setup()

                # Build complete pytest command
                pytest_args = self._build_pytest_command(
                    strategy, parallel_config
                )

                # Execute tests (this would integrate with pytest runner)
                raw_results = self._execute_pytest(pytest_args, strategy)

                # Analyze results
                result = strategy.analyze_results(raw_results)

                # Add performance data if available
                if performance_monitor:
                    output_path = None
                    if output_dir:
                        output_path = (
                            Path(output_dir)
                            / f"{strategy.name}_performance.json"
                        )

                    perf_report = (
                        performance_monitor.generate_consolidated_report(
                            output_path
                        )
                    )
                    result.performance_summary = {
                        "total_duration": perf_report.total_duration,
                        "efficiency_ratio": perf_report.efficiency_ratio,
                        "worker_count": perf_report.total_workers,
                        "aggregated_metrics": perf_report.aggregated_metrics,
                    }

                # Post-execution cleanup
                strategy.post_execution_cleanup(result)

                return result

            except Exception as e:
                self.logger.error(f"Strategy execution failed: {e}")
                return StrategyResult(
                    strategy_name=strategy.name,
                    strategy_type=strategy.strategy_type,
                    success=False,
                    execution_time=0.0,
                    tests_executed=0,
                    tests_passed=0,
                    tests_failed=0,
                    worker_count=0,
                    error_message=str(e),
                )
            finally:
                # Cleanup performance monitoring
                if performance_monitor:
                    self.performance_integration.cleanup_suite_monitor(
                        strategy.name
                    )

    def _build_pytest_command(
        self, strategy: BaseExecutionStrategy, config: ParallelTestConfig
    ) -> list[str]:
        """Build complete pytest command with all arguments.

        Args:
            strategy: Strategy being executed
            config: Parallel test configuration

        Returns:
            Complete pytest command arguments
        """
        command = ["pytest"]

        # Add strategy-specific arguments
        command.extend(strategy.get_pytest_args())

        # Add parallel execution arguments
        command.extend(config.to_pytest_args())

        # Add marker filters
        markers = strategy.get_markers()
        if markers:
            marker_expr = " and ".join(markers)
            command.extend(["-m", marker_expr])

        # Add test patterns
        command.extend(strategy.get_test_patterns())

        return command

    def _execute_pytest(
        self, pytest_args: list[str], strategy: BaseExecutionStrategy
    ) -> dict[str, Any]:
        """Execute pytest with given arguments.

        Args:
            pytest_args: Complete pytest command arguments
            strategy: Strategy being executed

        Returns:
            Raw execution results
        """
        import time

        start_time = time.time()

        try:
            # Note: In a real implementation, this would integrate with
            # pytest's API
            # or use subprocess to execute the actual pytest command
            self.logger.info(f"Would execute: {' '.join(pytest_args)}")

            # Simulate execution for demonstration
            execution_time = time.time() - start_time

            return {
                "success": True,
                "execution_time": execution_time,
                "tests_executed": 10,  # Mock data
                "tests_passed": 9,
                "tests_failed": 1,
                "worker_count": (
                    strategy.get_parallel_config().resource_limits.max_workers
                ),
                "command": pytest_args,
            }

        except Exception as e:
            execution_time = time.time() - start_time
            return {
                "success": False,
                "execution_time": execution_time,
                "tests_executed": 0,
                "tests_passed": 0,
                "tests_failed": 0,
                "worker_count": 0,
                "error_message": str(e),
                "command": pytest_args,
            }


# Factory functions for easy strategy creation
def create_smoke_strategy() -> SmokeTestStrategy:
    """Create a smoke test strategy."""
    return SmokeTestStrategy()


def create_performance_strategy() -> PerformanceTestStrategy:
    """Create a performance test strategy."""
    return PerformanceTestStrategy()


def create_full_strategy() -> FullTestStrategy:
    """Create a full test suite strategy."""
    return FullTestStrategy()


def create_development_strategy(
    focus_areas: list[str] | None = None,
) -> DevelopmentStrategy:
    """Create a development strategy.

    Args:
        focus_areas: Optional list of areas to focus on

    Returns:
        DevelopmentStrategy instance
    """
    return DevelopmentStrategy(focus_areas)


def create_ci_strategy() -> CIPipelineStrategy:
    """Create a CI pipeline strategy."""
    return CIPipelineStrategy()


def create_custom_strategy(
    name: str,
    pytest_args: list[str] | None = None,
    markers: list[str] | None = None,
    test_patterns: list[str] | None = None,
    monitor_performance: bool = False,
) -> CustomStrategy:
    """Create a custom strategy.

    Args:
        name: Name of the strategy
        pytest_args: Custom pytest arguments
        markers: Custom pytest markers
        test_patterns: Custom test file patterns
        monitor_performance: Whether to monitor performance

    Returns:
        CustomStrategy instance
    """
    return CustomStrategy(
        name=name,
        pytest_args=pytest_args,
        markers=markers,
        test_patterns=test_patterns,
        monitor_performance=monitor_performance,
    )


# Global executor instance
global_strategy_executor = StrategyExecutor()


def execute_strategy(
    strategy: BaseExecutionStrategy, output_dir: Path | str | None = None
) -> StrategyResult:
    """Execute a strategy using the global executor.

    Args:
        strategy: Strategy to execute
        output_dir: Optional output directory

    Returns:
        StrategyResult with execution results
    """
    return global_strategy_executor.execute_strategy(strategy, output_dir)
