"""Configuration file for pytest."""

import sys
from pathlib import Path
from typing import Any

import pytest

# Get the project root directory
project_root = Path(__file__).parent.parent

# Add the project root to sys.path to support absolute imports
sys.path.insert(0, str(project_root))


def pytest_configure(config: pytest.Config) -> None:
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "cuda: mark test as requiring CUDA capable device"
    )
    # Register hydra marker
    config.addinivalue_line(
        "markers", "hydra: mark test as using hydra for configuration"
    )
    # Register performance markers for performance testing integration
    config.addinivalue_line(
        "markers",
        "performance: mark test for performance measurement and monitoring",
    )
    config.addinivalue_line(
        "markers",
        "performance_critical: mark test as performance-critical with "
        "strict thresholds",
    )
    config.addinivalue_line(
        "markers",
        "performance_baseline: mark test for establishing performance "
        "baselines",
    )


# Added fixture
@pytest.fixture(scope="session")
def hydra_config_dir() -> str:
    """Provides the absolute path to the Hydra configuration directory."""
    # Path(__file__).parent is tests/
    # Path(__file__).parent.parent is the project root (crackseg/)
    project_root = Path(__file__).parent.parent
    config_path = project_root / "configs"
    return str(config_path.resolve())


@pytest.fixture(scope="session")
def performance_config() -> dict[str, Any]:
    """Provides performance testing configuration and thresholds.

    Returns:
        Dictionary containing performance thresholds and configuration
        for performance testing integration (subtask 15.7).
    """
    return {
        "thresholds": {
            "page_load_time_seconds": 5.0,  # Max acceptable page load time
            "interaction_latency_ms": 500.0,  # Max interaction response time
            "memory_usage_mb": 200.0,  # Max memory usage increase
            "training_start_seconds": 10.0,  # Max time to start training
        },
        "monitoring": {
            "enabled": True,
            "interval_seconds": 1.0,  # Memory monitoring interval
            "capture_browser_memory": True,
            "generate_reports": True,
        },
        "reporting": {
            "output_dir": "test-results/performance",
            "save_individual_reports": True,
            "generate_summary": True,
            "threshold_warnings": True,
        },
    }


@pytest.fixture
def performance_monitor_factory():
    """Factory fixture for creating PerformanceMonitor instances.

    Returns:
        Function that creates PerformanceMonitor instances with test names
    """
    from tests.e2e.helpers.performance_monitoring import PerformanceMonitor

    def create_monitor(test_name: str) -> PerformanceMonitor:
        """Create a PerformanceMonitor for the given test.

        Args:
            test_name: Name of the test for monitoring context

        Returns:
            Configured PerformanceMonitor instance
        """
        return PerformanceMonitor(test_name)

    return create_monitor


@pytest.fixture(autouse=True)
def performance_test_setup(request, performance_config: dict[str, Any]):
    """Auto-used fixture that sets up performance monitoring for marked tests.

    This fixture automatically activates for tests marked with
    @pytest.mark.performance, providing seamless performance integration
    without manual setup.
    """
    # Check if test is marked for performance monitoring
    performance_markers = [
        "performance",
        "performance_critical",
        "performance_baseline",
    ]

    is_performance_test = any(
        request.node.get_closest_marker(marker)
        for marker in performance_markers
    )

    if not is_performance_test:
        yield  # Test doesn't need performance monitoring
        return

    # Performance test setup
    test_name = f"{request.module.__name__}::{request.function.__name__}"

    # Create performance results directory
    output_dir = Path(performance_config["reporting"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Store performance context in request for use by test
    request.performance_config = performance_config
    request.performance_test_name = test_name
    request.performance_output_dir = output_dir

    yield

    # Performance test teardown - any cleanup if needed
