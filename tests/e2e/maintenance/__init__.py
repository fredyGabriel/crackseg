"""Test maintenance framework for comprehensive E2E testing.

This module provides a configurable framework for test review, maintenance,
and continuous improvement. It integrates seamlessly with existing reporting,
performance monitoring, and test infrastructure to provide automated
maintenance capabilities.

The framework supports:
- Test suite health monitoring with automated alerts
- Configurable maintenance cycles and scheduling
- Performance optimization recommendations and execution
- Continuous improvement tracking and metrics
- Integration with existing reporting and capture systems
"""

from tests.e2e.maintenance.config import (
    MaintenanceConfig,
    MaintenanceMode,
    ReviewFrequency,
)
from tests.e2e.maintenance.core import TestMaintenanceManager
from tests.e2e.maintenance.health_monitor import TestSuiteHealthMonitor
from tests.e2e.maintenance.models import (
    HealthStatus,
    MaintenanceReport,
    ReviewResult,
)
from tests.e2e.maintenance.optimization import PerformanceOptimizer
from tests.e2e.maintenance.review_cycles import AutomatedReviewCycles
from tests.e2e.maintenance.scheduling import MaintenanceScheduler

__all__ = [
    # Core classes
    "TestMaintenanceManager",
    "TestSuiteHealthMonitor",
    "AutomatedReviewCycles",
    "PerformanceOptimizer",
    "MaintenanceScheduler",
    # Configuration
    "MaintenanceConfig",
    "MaintenanceMode",
    "ReviewFrequency",
    # Models
    "HealthStatus",
    "MaintenanceReport",
    "ReviewResult",
]

# Version and metadata
__version__ = "1.0.0"
__author__ = "CrackSeg Team"
__description__ = "Configurable test maintenance framework"


def create_maintenance_manager(
    config: MaintenanceConfig | None = None,
) -> TestMaintenanceManager:
    """Create a configured test maintenance manager.

    Args:
        config: Optional maintenance configuration

    Returns:
        Configured TestMaintenanceManager instance

    Example:
        >>> manager = create_maintenance_manager()
        >>> health_status = manager.check_suite_health()
        >>> if health_status.requires_maintenance:
        ...     manager.run_maintenance_cycle()
    """
    return TestMaintenanceManager(config or MaintenanceConfig())


def quick_health_check() -> HealthStatus:
    """Perform a quick health check of the test suite.

    Returns:
        HealthStatus with basic health metrics

    Example:
        >>> status = quick_health_check()
        >>> print(f"Suite health: {status.overall_health}")
    """
    monitor = TestSuiteHealthMonitor()
    return monitor.quick_health_check()
