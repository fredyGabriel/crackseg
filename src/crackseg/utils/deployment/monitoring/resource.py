"""Resource monitoring for deployment system."""

import logging
import time
from typing import TYPE_CHECKING

import psutil

if TYPE_CHECKING:
    from .config import ResourceMetrics


class ResourceMonitor:
    """Monitor system and process resources."""

    def __init__(self) -> None:
        """Initialize resource monitor."""
        self.logger = logging.getLogger(__name__)

    def get_system_metrics(self) -> "ResourceMetrics":
        """Get current system resource metrics.

        Returns:
            System resource metrics
        """
        try:
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            network = psutil.net_io_counters()

            # Calculate network I/O in Mbps
            network_io_mbps = (
                (network.bytes_sent + network.bytes_recv) * 8 / 1_000_000
            )

            return ResourceMetrics(
                cpu_usage_percent=cpu_usage,
                memory_usage_mb=memory.used / 1024 / 1024,  # Convert to MB
                disk_usage_percent=disk.percent,
                network_io_mbps=network_io_mbps,
                timestamp=time.time(),
            )

        except Exception as e:
            self.logger.error(f"Failed to get system metrics: {e}")
            return ResourceMetrics(
                cpu_usage_percent=0.0,
                memory_usage_mb=0.0,
                disk_usage_percent=0.0,
                network_io_mbps=0.0,
                timestamp=time.time(),
            )

    def get_process_metrics(self, process_name: str) -> "ResourceMetrics":
        """Get metrics for a specific process.

        Args:
            process_name: Name of the process to monitor

        Returns:
            Process resource metrics
        """
        try:
            for proc in psutil.process_iter(
                ["pid", "name", "cpu_percent", "memory_info"]
            ):
                if (
                    proc.info["name"]
                    and process_name.lower() in proc.info["name"].lower()
                ):
                    # Get process-specific metrics
                    cpu_usage = proc.info["cpu_percent"]
                    memory_info = proc.info["memory_info"]
                    memory_mb = memory_info.rss / 1024 / 1024  # Convert to MB

                    return ResourceMetrics(
                        cpu_usage_percent=cpu_usage,
                        memory_usage_mb=memory_mb,
                        disk_usage_percent=0.0,  # Process disk usage not easily available
                        network_io_mbps=0.0,  # Process network usage not easily available
                        timestamp=time.time(),
                    )

            # Process not found
            self.logger.warning(f"Process '{process_name}' not found")
            return ResourceMetrics(
                cpu_usage_percent=0.0,
                memory_usage_mb=0.0,
                disk_usage_percent=0.0,
                network_io_mbps=0.0,
                timestamp=time.time(),
            )

        except Exception as e:
            self.logger.error(f"Failed to get process metrics: {e}")
            return ResourceMetrics(
                cpu_usage_percent=0.0,
                memory_usage_mb=0.0,
                disk_usage_percent=0.0,
                network_io_mbps=0.0,
                timestamp=time.time(),
            )

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB.

        Returns:
            Memory usage in MB
        """
        try:
            memory = psutil.virtual_memory()
            return memory.used / 1024 / 1024  # Convert to MB
        except Exception as e:
            self.logger.error(f"Failed to get memory usage: {e}")
            return 0.0

    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage.

        Returns:
            CPU usage percentage
        """
        try:
            return psutil.cpu_percent(interval=1)
        except Exception as e:
            self.logger.error(f"Failed to get CPU usage: {e}")
            return 0.0

    def get_disk_usage(self, path: str = "/") -> float:
        """Get disk usage percentage for a path.

        Args:
            path: Path to check disk usage for

        Returns:
            Disk usage percentage
        """
        try:
            disk = psutil.disk_usage(path)
            return disk.percent
        except Exception as e:
            self.logger.error(f"Failed to get disk usage: {e}")
            return 0.0
