"""Health check enumeration types."""

from enum import Enum


class HealthStatus(Enum):
    """Health status enumeration."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    UNKNOWN = "unknown"
    CRITICAL = "critical"
