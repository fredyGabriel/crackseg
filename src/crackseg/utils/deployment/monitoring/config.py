"""Configuration dataclasses for monitoring system."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HealthCheckConfig:
    """Configuration for health checks."""

    url: str
    timeout: int = 10
    interval: int = 30
    max_retries: int = 3
    success_threshold: int = 1
    failure_threshold: int = 3


@dataclass
class MetricsConfig:
    """Configuration for metrics collection."""

    collection_interval: int = 30
    retention_period: int = 3600  # 1 hour
    max_data_points: int = 1000
    enabled_metrics: list[str] = field(
        default_factory=lambda: [
            "response_time",
            "throughput",
            "error_rate",
            "memory_usage",
            "cpu_usage",
        ]
    )


@dataclass
class DashboardConfig:
    """Configuration for monitoring dashboard."""

    dashboard_url: str = ""
    refresh_interval: int = 30
    auto_refresh: bool = True
    theme: str = "light"
    custom_panels: dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertThresholds:
    """Thresholds for alerting."""

    response_time_ms: float = 1000.0
    error_rate_percent: float = 5.0
    memory_usage_percent: float = 80.0
    cpu_usage_percent: float = 80.0
    disk_usage_percent: float = 90.0


@dataclass
class MonitoringResult:
    """Result of monitoring operation."""

    success: bool
    timestamp: float
    metrics: dict[str, Any] = field(default_factory=dict)
    health_status: str = "unknown"
    alerts: list[str] = field(default_factory=list)
    error_message: str | None = None


@dataclass
class ResourceMetrics:
    """System resource metrics."""

    cpu_usage_percent: float
    memory_usage_mb: float
    disk_usage_percent: float
    network_io_mbps: float
    timestamp: float
