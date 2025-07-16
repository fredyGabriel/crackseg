"""
Monitoring framework for collecting performance and system metrics.

This package provides a callback-based monitoring system that can be
integrated with training loops to collect various metrics like timing,
system resource usage, and GPU statistics.

Includes resource monitoring and alerting system (Subtask 16.4).
"""

from .alert_types import Alert, AlertCallback, AlertSeverity, AlertType
from .alerting_system import AlertingSystem
from .callbacks import BaseCallback, CallbackHandler, TimerCallback
from .exceptions import MonitoringError
from .gpu_callbacks import GPUStatsCallback
from .manager import MonitoringManager
from .resource_monitor import ResourceMonitor
from .resource_snapshot import ResourceDict, ResourceSnapshot
from .retention import (
    CompositeRetentionPolicy,
    CountBasedRetentionPolicy,
    RetentionManager,
    RetentionPolicy,
    TimeBasedRetentionPolicy,
)
from .system_callbacks import SystemStatsCallback
from .threshold_checker import ThresholdChecker
from .threshold_config import ThresholdConfig

__all__ = [
    # Original monitoring framework
    "BaseCallback",
    "CallbackHandler",
    "TimerCallback",
    "MonitoringManager",
    "SystemStatsCallback",
    "GPUStatsCallback",
    "MonitoringError",
    "RetentionPolicy",
    "TimeBasedRetentionPolicy",
    "CountBasedRetentionPolicy",
    "CompositeRetentionPolicy",
    "RetentionManager",
    # Resource monitoring (Subtask 16.4)
    "ResourceMonitor",
    "ResourceSnapshot",
    "ResourceDict",
    # Alerting system (Subtask 16.4)
    "AlertingSystem",
    "Alert",
    "AlertType",
    "AlertSeverity",
    "AlertCallback",
    "ThresholdChecker",
    "ThresholdConfig",
]
