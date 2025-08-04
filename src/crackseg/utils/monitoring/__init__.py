"""
Monitoring framework for collecting performance and system metrics.

This package provides a callback-based monitoring system that can be
integrated with training loops to collect various metrics like timing,
system resource usage, and GPU statistics.

Includes resource monitoring and alerting system (Subtask 16.4).
"""

# Import from new modular structure
from .alerts import (
    Alert,
    AlertCallback,
    AlertingSystem,
    AlertSeverity,
    AlertType,
    ThresholdChecker,
)
from .callbacks import (
    BaseCallback,
    CallbackHandler,
    GPUStatsCallback,
    SystemStatsCallback,
    TimerCallback,
)
from .coverage import CoverageMonitor
from .exceptions import MonitoringError
from .manager import MonitoringManager
from .resources import (
    ResourceDict,
    ResourceMonitor,
    ResourceSnapshot,
    ThresholdConfig,
)
from .retention import (
    CompositeRetentionPolicy,
    CountBasedRetentionPolicy,
    RetentionManager,
    RetentionPolicy,
    TimeBasedRetentionPolicy,
)

__all__ = [
    # Core monitoring framework
    "MonitoringManager",
    "MonitoringError",
    # Callback system
    "BaseCallback",
    "CallbackHandler",
    "TimerCallback",
    "SystemStatsCallback",
    "GPUStatsCallback",
    # Resource monitoring
    "ResourceMonitor",
    "ResourceSnapshot",
    "ResourceDict",
    "ThresholdConfig",
    # Alerting system
    "AlertingSystem",
    "Alert",
    "AlertType",
    "AlertSeverity",
    "AlertCallback",
    "ThresholdChecker",
    # Retention policies
    "RetentionPolicy",
    "TimeBasedRetentionPolicy",
    "CountBasedRetentionPolicy",
    "CompositeRetentionPolicy",
    "RetentionManager",
    # Coverage monitoring
    "CoverageMonitor",
]
