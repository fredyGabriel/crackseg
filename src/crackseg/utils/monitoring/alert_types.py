"""Alert types and data structures for resource monitoring.

This module defines alert types, severity levels, and the Alert dataclass
used by the alerting system for crack segmentation monitoring.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertType(Enum):
    """Types of resource alerts."""

    # System resource alerts
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    MEMORY_LEAK = "memory_leak"

    # GPU alerts (RTX 3070 Ti specific)
    GPU_MEMORY_USAGE = "gpu_memory_usage"
    GPU_UTILIZATION = "gpu_utilization"
    GPU_TEMPERATURE = "gpu_temperature"
    GPU_OUT_OF_MEMORY = "gpu_out_of_memory"

    # Process alerts
    PROCESS_COUNT = "process_count"
    FILE_HANDLES = "file_handles"
    THREAD_COUNT = "thread_count"

    # Application alerts
    TEMP_FILES_ACCUMULATION = "temp_files_accumulation"
    DISK_SPACE = "disk_space"
    NETWORK_CONNECTIONS = "network_connections"

    # Performance alerts
    RESPONSE_TIME = "response_time"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


@dataclass
class Alert:
    """Resource monitoring alert."""

    alert_type: AlertType
    severity: AlertSeverity
    message: str
    value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.now)
    resource_name: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_timestamp: datetime | None = None

    def resolve(self) -> None:
        """Mark alert as resolved."""
        self.resolved = True
        self.resolution_timestamp = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert alert to dictionary for logging/reporting."""
        return {
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "value": self.value,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat(),
            "resource_name": self.resource_name,
            "context": self.context,
            "resolved": self.resolved,
            "resolution_timestamp": (
                self.resolution_timestamp.isoformat()
                if self.resolution_timestamp
                else None
            ),
        }


# Type alias for callback functions
type AlertCallback = Callable[[Alert], None]
