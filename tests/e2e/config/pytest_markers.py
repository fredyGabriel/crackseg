"""
Pytest markers configuration for parallel execution and performance
testing. This module defines custom pytest markers that enable
fine-grained control over test execution strategies, resource
allocation, and performance monitoring within the parallel test
execution framework.
"""

from typing import Any

import pytest


# MarkDecorator is not part of pytest's public API; use Any for decorators
def pytest_configure(config: Any) -> None:
    """
    Register custom markers for parallel execution and performance
    testing. Args: config: Pytest configuration object
    """
    # Execution strategy markers
    config.addinivalue_line(
        "markers",
        "sequential: Run test sequentially (disable parallelization)",
    )
    config.addinivalue_line(
        "markers", "parallel: Run test with parallel execution (default)"
    )
    config.addinivalue_line(
        "markers",
        "distributed: Run test with distributed execution across nodes",
    )
    config.addinivalue_line(
        "markers",
        "browser_matrix: Run test across multiple browser configurations",
    )

    # Performance monitoring markers
    config.addinivalue_line(
        "markers", "performance: Enable basic performance monitoring for test"
    )
    config.addinivalue_line(
        "markers",
        "performance_critical: Enable detailed performance monitoring "
        "with strict thresholds",
    )
    config.addinivalue_line(
        "markers",
        "performance_baseline: Establish performance baseline for "
        "regression detection",
    )
    config.addinivalue_line(
        "markers",
        "performance_regression: Test for performance regression detection",
    )

    # Resource allocation markers
    config.addinivalue_line(
        "markers",
        "high_memory: Test requires high memory allocation (>4GB per worker)",
    )
    config.addinivalue_line(
        "markers",
        "low_memory: Test can run with minimal memory (<1GB per worker)",
    )
    config.addinivalue_line(
        "markers", "cpu_intensive: Test requires significant CPU resources"
    )
    config.addinivalue_line(
        "markers",
        "io_intensive: Test is I/O bound (optimize for concurrent execution)",
    )

    # Test isolation markers
    config.addinivalue_line(
        "markers", "isolated: Test requires complete isolation (run alone)"
    )
    config.addinivalue_line(
        "markers", "shared_state: Test can share state with other tests"
    )
    config.addinivalue_line(
        "markers", "stateful: Test maintains state between invocations"
    )

    # Execution priority markers
    config.addinivalue_line(
        "markers",
        "priority_high: High priority test (run first in parallel queue)",
    )
    config.addinivalue_line(
        "markers", "priority_low: Low priority test (run after others)"
    )
    config.addinivalue_line(
        "markers",
        "critical_path: Test is on critical path (optimize for speed)",
    )

    # Environment and configuration markers
    config.addinivalue_line(
        "markers", "docker_only: Test requires Docker environment"
    )
    config.addinivalue_line(
        "markers", "local_only: Test only runs in local environment (not CI)"
    )
    config.addinivalue_line(
        "markers", "ci_only: Test only runs in CI environment"
    )


# Marker helper functions for easier test decoration
def sequential(reason: str = "Requires sequential execution") -> Any:
    """
    Mark test for sequential execution. Args: reason: Reason why test
    needs sequential execution Returns: Pytest marker decorator
    """
    return pytest.mark.sequential(reason=reason)


def parallel(workers: int | None = None, strategy: str | None = None) -> Any:
    """
    Mark test for parallel execution with optional configuration. Args:
    workers: Number of workers to use for this test strategy: Execution
    strategy override Returns: Pytest marker decorator
    """
    kwargs: dict[str, Any] = {}
    if workers is not None:
        kwargs["workers"] = workers
    if strategy is not None:
        kwargs["strategy"] = strategy

    return pytest.mark.parallel(**kwargs)


def performance(
    threshold_seconds: float = 30.0,
    memory_threshold_mb: float = 1024.0,
    regression_check: bool = False,
) -> Any:
    """
    Mark test for performance monitoring. Args: threshold_seconds: Maximum
    acceptable execution time memory_threshold_mb: Maximum acceptable
    memory usage regression_check: Enable regression detection Returns:
    Pytest marker decorator
    """
    return pytest.mark.performance(
        threshold_seconds=threshold_seconds,
        memory_threshold_mb=memory_threshold_mb,
        regression_check=regression_check,
    )


def performance_critical(
    threshold_seconds: float = 10.0, strict_monitoring: bool = True
) -> Any:
    """
    Mark test for critical performance monitoring. Args:
    threshold_seconds: Strict performance threshold strict_monitoring:
    Enable detailed monitoring Returns: Pytest marker decorator
    """
    return pytest.mark.performance_critical(
        threshold_seconds=threshold_seconds,
        strict_monitoring=strict_monitoring,
    )


def performance_baseline(baseline_name: str) -> Any:
    """
    Mark test as performance baseline. Args: baseline_name: Name of the
    performance baseline Returns: Pytest marker decorator
    """
    return pytest.mark.performance_baseline(baseline_name=baseline_name)


def browser_matrix(
    browsers: list[str] | None = None, parallel_browsers: bool = True
) -> Any:
    """Mark test for cross-browser matrix execution.

    Args:
        browsers: List of browsers to test (default: ["chrome", "firefox"])
        parallel_browsers: Run browsers in parallel

    Returns:
        Pytest marker decorator
    """
    if browsers is None:
        browsers = ["chrome", "firefox"]

    return pytest.mark.browser_matrix(
        browsers=browsers, parallel_browsers=parallel_browsers
    )


def resource_requirement(
    memory_mb: int, cpu_cores: float = 1.0, isolated: bool = False
) -> Any:
    """
    Mark test with specific resource requirements. Args: memory_mb:
    Required memory in MB cpu_cores: Required CPU cores (can be
    fractional) isolated: Whether test needs isolation Returns: Pytest
    marker decorator
    """
    markers: list[Any] = []

    if memory_mb > 4096:
        markers.append(pytest.mark.high_memory)
    elif memory_mb < 1024:
        markers.append(pytest.mark.low_memory)

    if cpu_cores > 2.0:
        markers.append(pytest.mark.cpu_intensive)

    if isolated:
        markers.append(pytest.mark.isolated)

    # Return composite marker if multiple, single marker otherwise
    if len(markers) == 1:
        return markers[0]
    else:
        # Create a composite marker
        return pytest.mark.resource_requirement(
            memory_mb=memory_mb, cpu_cores=cpu_cores, isolated=isolated
        )


def priority(level: str, reason: str = "") -> Any:
    """Mark test with execution priority.

    Args:
        level: Priority level ("high", "normal", "low")
        reason: Reason for priority assignment

    Returns:
        Pytest marker decorator
    """
    if level == "high":
        return pytest.mark.priority_high(reason=reason)
    elif level == "low":
        return pytest.mark.priority_low(reason=reason)
    else:
        return pytest.mark.priority_normal(reason=reason)


def environment(env_type: str) -> Any:
    """Mark test for specific environment.

    Args:
        env_type: Environment type ("docker", "local", "ci")

    Returns:
        Pytest marker decorator
    """
    if env_type == "docker":
        return pytest.mark.docker_only
    elif env_type == "local":
        return pytest.mark.local_only
    elif env_type == "ci":
        return pytest.mark.ci_only
    else:
        return pytest.mark.any_environment


# Utility functions for test configuration
def get_test_markers(item: Any) -> dict[str, Any]:
    """
    Extract marker information from a test item. Args: item: Pytest test
    item Returns: Dictionary of marker information
    """
    markers_info: dict[str, Any] = {}

    for marker in item.iter_markers():
        markers_info[marker.name] = {
            "args": marker.args,
            "kwargs": marker.kwargs,
        }

    return markers_info


def should_run_parallel(item: Any) -> bool:
    """
    Determine if test should run in parallel based on markers. Args: item:
    Pytest test item Returns: True if test should run in parallel
    """
    # Check for explicit sequential marker
    if item.get_closest_marker("sequential"):
        return False

    # Check for isolation requirement
    if item.get_closest_marker("isolated"):
        return False

    # Default to parallel unless explicitly marked otherwise
    return True


def get_worker_count_for_test(item: Any) -> int | None:
    """
    Get optimal worker count for a specific test. Args: item: Pytest test
    item Returns: Recommended worker count or None for default
    """
    markers_info = get_test_markers(item)

    # Check for explicit worker specification
    if "parallel" in markers_info:
        return markers_info["parallel"].get("kwargs", {}).get("workers")

    # Check resource requirements
    if "high_memory" in markers_info:
        return 2  # Limit workers for high memory tests

    if "cpu_intensive" in markers_info:
        return 4  # More workers for CPU intensive tests

    if "io_intensive" in markers_info:
        return 8  # Many workers for I/O bound tests

    return None  # Use default


def get_performance_config_for_test(item: Any) -> dict[str, Any]:
    """
    Get performance monitoring configuration for a test. Args: item:
    Pytest test item Returns: Performance configuration dictionary
    """
    config = {
        "enabled": False,
        "threshold_seconds": 30.0,
        "memory_threshold_mb": 1024.0,
        "regression_check": False,
        "detailed_monitoring": False,
    }

    markers_info = get_test_markers(item)

    # Basic performance monitoring
    if "performance" in markers_info:
        config["enabled"] = True
        perf_kwargs = markers_info["performance"].get("kwargs", {})
        config.update(perf_kwargs)

    # Critical performance monitoring
    if "performance_critical" in markers_info:
        config["enabled"] = True
        config["detailed_monitoring"] = True
        config["threshold_seconds"] = 10.0
        crit_kwargs = markers_info["performance_critical"].get("kwargs", {})
        config.update(crit_kwargs)

    # Baseline establishment
    if "performance_baseline" in markers_info:
        config["enabled"] = True
        config["is_baseline"] = True
        baseline_kwargs = markers_info["performance_baseline"].get(
            "kwargs", {}
        )
        config.update(baseline_kwargs)

    # Regression testing
    if "performance_regression" in markers_info:
        config["enabled"] = True
        config["regression_check"] = True
        regr_kwargs = markers_info["performance_regression"].get("kwargs", {})
        config.update(regr_kwargs)

    return config


def should_enable_performance_monitoring(item: Any) -> bool:
    """
    Check if performance monitoring should be enabled for a test. Args:
    item: Pytest test item Returns: True if performance monitoring should
    be enabled
    """
    performance_markers = [
        "performance",
        "performance_critical",
        "performance_baseline",
        "performance_regression",
    ]

    return any(
        item.get_closest_marker(marker) for marker in performance_markers
    )


def get_marker_configuration() -> dict[str, Any]:
    """
    Get general marker configuration for tests. Returns: Dictionary with
    marker configuration settings
    """
    return {
        "parallel_enabled": True,
        "performance_monitoring": True,
        "resource_tracking": True,
        "default_workers": 4,
        "timeout_seconds": 300,
    }


def get_performance_markers() -> dict[str, Any]:
    """
    Get available performance-related markers. Returns: Dictionary of
    performance markers and their configurations
    """
    return {
        "performance": {
            "threshold_seconds": 30.0,
            "memory_threshold_mb": 1024.0,
            "regression_check": False,
        },
        "performance_critical": {
            "threshold_seconds": 10.0,
            "strict_monitoring": True,
        },
        "performance_baseline": {
            "baseline_name": "default",
        },
    }


def get_resource_markers() -> dict[str, Any]:
    """
    Get available resource-related markers. Returns: Dictionary of
    resource markers and their configurations
    """
    return {
        "resource_requirement": {
            "memory_mb": 512,
            "cpu_cores": 1.0,
            "isolated": False,
        },
        "browser_matrix": {
            "browsers": ["chrome", "firefox"],
            "parallel_browsers": True,
        },
        "environment": {
            "env_type": "test",
        },
    }


# Export all marker functions and utilities
__all__ = [
    "pytest_configure",
    "sequential",
    "parallel",
    "performance",
    "performance_critical",
    "performance_baseline",
    "browser_matrix",
    "resource_requirement",
    "priority",
    "environment",
    "get_test_markers",
    "should_run_parallel",
    "get_worker_count_for_test",
    "get_performance_config_for_test",
    "should_enable_performance_monitoring",
    "get_marker_configuration",
    "get_performance_markers",
    "get_resource_markers",
]
