"""Pytest plugin for automatic test performance optimization.

This plugin integrates the performance optimization framework directly into
pytest execution, providing seamless optimization without manual configuration.
Part of subtask 7.5 - Test Execution Performance Optimization.
"""

import pytest

from .performance_optimizer import (
    OptimizationConfig,
    TestPerformanceOptimizer,
    get_performance_optimizer,
)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add command line options for performance optimization.

    Args:
        parser: Pytest argument parser
    """
    group = parser.getgroup("performance", "Test performance optimization")

    group.addoption(
        "--performance-optimize",
        action="store_true",
        default=False,
        help="Enable automatic performance optimizations",
    )

    group.addoption(
        "--fixture-cache",
        action="store_true",
        default=True,
        help="Enable fixture caching (default: True)",
    )

    group.addoption(
        "--no-fixture-cache",
        action="store_false",
        dest="fixture_cache",
        help="Disable fixture caching",
    )

    group.addoption(
        "--selective-tests",
        action="store_true",
        default=False,
        help="Enable selective test running based on changes",
    )

    group.addoption(
        "--parallel-auto",
        action="store_true",
        default=False,
        help="Automatically determine optimal parallel worker count",
    )

    group.addoption(
        "--performance-baseline",
        action="store_true",
        default=False,
        help="Establish performance baseline for regression detection",
    )

    group.addoption(
        "--performance-report",
        type=str,
        metavar="FILE",
        help="Generate performance report to specified file",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with performance optimization settings.

    Args:
        config: Pytest configuration object
    """
    # Register performance markers
    config.addinivalue_line(
        "markers",
        "performance_test: Mark test for performance optimization tracking",
    )
    config.addinivalue_line(
        "markers",
        "expensive_fixture: Mark fixture as expensive (should be cached)",
    )
    config.addinivalue_line(
        "markers", "fast_test: Mark test as fast (minimal overhead)"
    )

    # Configure optimization if enabled
    if config.getoption("--performance-optimize"):
        optimization_config = OptimizationConfig(
            enable_fixture_caching=config.getoption("--fixture-cache"),
            enable_selective_running=config.getoption("--selective-tests"),
            auto_parallel_detection=config.getoption("--parallel-auto"),
            performance_baseline_enabled=config.getoption(
                "--performance-baseline"
            ),
            metrics_collection_enabled=True,
        )

        # Store config in pytest config for use by other hooks
        config._performance_optimizer = TestPerformanceOptimizer(
            optimization_config
        )


@pytest.fixture(scope="session", autouse=True)
def performance_optimization_session(request: pytest.FixtureRequest):
    """Session-scoped fixture to set up performance optimization.

    Args:
        request: Pytest fixture request
    """
    if hasattr(request.config, "_performance_optimizer"):
        optimizer = request.config._performance_optimizer
        print(f"Performance optimization enabled: {optimizer.config}")


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Hook called before each test setup.

    Args:
        item: Test item being executed
    """
    # Apply optimizations based on test markers
    if item.get_closest_marker("expensive_fixture"):
        # Ensure expensive fixtures are cached
        optimizer = get_performance_optimizer()
        if not optimizer.config.enable_fixture_caching:
            pytest.skip("Expensive fixture test requires caching enabled")


def pytest_runtest_call(item: pytest.Item) -> None:
    """Hook called during test execution.

    Args:
        item: Test item being executed
    """
    # Add performance tracking for marked tests
    if item.get_closest_marker("performance_test"):
        # This would integrate with performance monitoring
        pass


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter,
    exitstatus: int,
    config: pytest.Config,
) -> None:
    """Hook to display performance summary at the end of test run.

    Args:
        terminalreporter: Terminal reporter for output
        exitstatus: Exit status of test run
        config: Pytest configuration
    """
    if hasattr(config, "_performance_optimizer"):
        optimizer = config._performance_optimizer

        # Display optimization summary
        terminalreporter.write_sep("=", "Performance Optimization Summary")

        if optimizer.config.enable_fixture_caching:
            terminalreporter.write_line("✓ Fixture caching enabled")

        if optimizer.config.enable_selective_running:
            terminalreporter.write_line("✓ Selective test running enabled")

        if optimizer.config.auto_parallel_detection:
            worker_count = (
                optimizer.parallel_optimizer.get_optimal_worker_count()
            )
            terminalreporter.write_line(
                f"✓ Auto-parallel detection: {worker_count} workers"
            )

        # Generate performance report if requested
        report_file = config.getoption("--performance-report")
        if report_file:
            _generate_performance_report(optimizer, report_file)
            terminalreporter.write_line(
                f"Performance report saved to: {report_file}"
            )


def _generate_performance_report(
    optimizer: TestPerformanceOptimizer, report_file: str
) -> None:
    """Generate a performance report file.

    Args:
        optimizer: Performance optimizer instance
        report_file: Path to report file
    """
    import json
    from pathlib import Path

    report_data = {
        "optimization_config": {
            "fixture_caching": optimizer.config.enable_fixture_caching,
            "selective_running": optimizer.config.enable_selective_running,
            "auto_parallel": optimizer.config.auto_parallel_detection,
            "performance_baseline": (
                optimizer.config.performance_baseline_enabled
            ),
        },
        "metrics": [
            {
                "test_name": metric.test_name,
                "execution_time": metric.execution_time,
                "setup_time": metric.setup_time,
                "fixture_cache_hits": metric.fixture_cache_hits,
                "fixture_cache_misses": metric.fixture_cache_misses,
                "parallel_efficiency": metric.parallel_efficiency,
                "memory_usage_mb": metric.memory_usage_mb,
            }
            for metric in optimizer.metrics
        ],
    }

    Path(report_file).write_text(json.dumps(report_data, indent=2))
