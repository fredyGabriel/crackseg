"""Health check result data models."""

from datetime import datetime
from typing import Any

from crackseg.dataclasses import dataclass

from .enums import HealthStatus


@dataclass
class HealthCheckResult:
    """Result of a health check operation."""

    service_name: str
    status: HealthStatus
    response_time: float
    timestamp: datetime
    details: dict[str, Any]
    error_message: str | None = None


@dataclass
class SystemHealthReport:
    """Comprehensive system health report."""

    timestamp: datetime
    overall_status: HealthStatus
    services: dict[str, HealthCheckResult]
    dependencies_satisfied: bool
    recommendations: list[str]
    metrics: dict[str, Any]
