"""Resource snapshot data structures and utilities.

This module contains data structures for capturing system resource states
at specific points in time, supporting the crack segmentation monitoring.
"""

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Type aliases for clarity
type ResourceDict = dict[str, float | int | bool]


@dataclass
class ResourceSnapshot:
    """Snapshot of system resources at a specific point in time."""

    timestamp: float

    # System resources
    cpu_percent: float
    memory_used_mb: float
    memory_available_mb: float
    memory_percent: float

    # GPU resources (RTX 3070 Ti specific)
    gpu_memory_used_mb: float
    gpu_memory_total_mb: float
    gpu_memory_percent: float
    gpu_utilization_percent: float
    gpu_temperature_celsius: float

    # Process resources
    process_count: int
    thread_count: int
    file_handles: int

    # Network resources
    network_connections: int
    open_ports: list[int]

    # Disk I/O
    disk_read_mb: float
    disk_write_mb: float

    # Application-specific
    temp_files_count: int
    temp_files_size_mb: float

    def to_dict(self) -> ResourceDict:
        """Convert snapshot to dictionary format."""
        return {
            "timestamp": self.timestamp,
            "cpu_percent": self.cpu_percent,
            "memory_used_mb": self.memory_used_mb,
            "memory_available_mb": self.memory_available_mb,
            "memory_percent": self.memory_percent,
            "gpu_memory_used_mb": self.gpu_memory_used_mb,
            "gpu_memory_total_mb": self.gpu_memory_total_mb,
            "gpu_memory_percent": self.gpu_memory_percent,
            "gpu_utilization_percent": self.gpu_utilization_percent,
            "gpu_temperature_celsius": self.gpu_temperature_celsius,
            "process_count": self.process_count,
            "thread_count": self.thread_count,
            "file_handles": self.file_handles,
            "network_connections": self.network_connections,
            "open_ports": len(self.open_ports),
            "disk_read_mb": self.disk_read_mb,
            "disk_write_mb": self.disk_write_mb,
            "temp_files_count": self.temp_files_count,
            "temp_files_size_mb": self.temp_files_size_mb,
        }

    def get_summary(self) -> dict[str, Any]:
        """Get human-readable summary of resource usage."""
        memory_usage = (
            f"{self.memory_used_mb:.0f}MB ({self.memory_percent:.1f}%)"
        )
        gpu_memory = (
            f"{self.gpu_memory_used_mb:.0f}MB/"
            f"{self.gpu_memory_total_mb:.0f}MB "
            f"({self.gpu_memory_percent:.1f}%)"
        )
        temp_files = (
            f"{self.temp_files_count} files ({self.temp_files_size_mb:.1f}MB)"
        )

        return {
            "cpu_usage": f"{self.cpu_percent:.1f}%",
            "memory_usage": memory_usage,
            "gpu_memory": gpu_memory,
            "gpu_utilization": f"{self.gpu_utilization_percent:.1f}%",
            "gpu_temperature": f"{self.gpu_temperature_celsius:.1f}°C",
            "processes": self.process_count,
            "threads": self.thread_count,
            "file_handles": self.file_handles,
            "network_connections": self.network_connections,
            "temp_files": temp_files,
        }
