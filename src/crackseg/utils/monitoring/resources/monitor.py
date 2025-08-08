"""Real-time resource monitoring for comprehensive system tracking.

This module provides unified resource monitoring capabilities that integrate
with the existing MonitoringManager framework, supporting real-time tracking
of CPU, memory, disk, network, GPU, and application-specific resources.

Designed specifically for crack segmentation workflows with RTX 3070 Ti
optimization and E2E testing requirements.
"""

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import psutil

from ..exceptions import MonitoringError
from ..manager import MonitoringManager
from .snapshot import ResourceDict, ResourceSnapshot

logger = logging.getLogger(__name__)

# Type aliases for clarity
type ResourceCallback = Callable[[ResourceDict], None]


class ResourceMonitor:
    """Unified resource monitoring with real-time tracking capabilities.

    Integrates all existing monitoring components and provides real-time
    resource tracking with configurable sampling intervals and callbacks.

    Features:
    - Real-time system resource monitoring
    - GPU monitoring (RTX 3070 Ti optimized)
    - Process lifecycle tracking
    - Network connection monitoring
    - File handle tracking
    - Integration with MonitoringManager
    - Configurable alerting thresholds

    Example:
        >>> monitor = ResourceMonitor(monitoring_manager)
        >>> monitor.start_real_time_monitoring(interval=1.0)
        >>> snapshot = monitor.get_current_snapshot()
        >>> monitor.stop_monitoring()
    """

    def __init__(
        self,
        monitoring_manager: MonitoringManager | None = None,
        enable_gpu_monitoring: bool = True,
        enable_network_monitoring: bool = True,
        enable_file_monitoring: bool = True,
    ) -> None:
        """Initialize the resource monitor."""
        self.monitoring_manager = monitoring_manager
        self.enable_gpu_monitoring = enable_gpu_monitoring
        self.enable_network_monitoring = enable_network_monitoring
        self.enable_file_monitoring = enable_file_monitoring

        # Real-time monitoring state
        self._monitoring_active = False
        self._monitoring_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # Resource tracking
        self._snapshots: list[ResourceSnapshot] = []
        self._max_snapshots = 1000  # Limit memory usage
        self._callbacks: list[ResourceCallback] = []

        # Performance optimization
        self._disk_io_baseline: dict[str, Any] | None = None
        self._network_baseline: dict[str, Any] | None = None

        # Initialize baselines
        self._initialize_baselines()

    def _initialize_baselines(self) -> None:
        """Initialize performance baselines for delta calculations."""
        try:
            disk_io = psutil.disk_io_counters()
            network_io = psutil.net_io_counters()
            self._disk_io_baseline = disk_io._asdict() if disk_io else None
            self._network_baseline = (
                network_io._asdict() if network_io else None
            )
        except (AttributeError, OSError):
            logger.warning("Could not initialize I/O baselines")

    def add_callback(self, callback: ResourceCallback) -> None:
        """Add callback to be executed on each resource update."""
        with self._lock:
            self._callbacks.append(callback)

    def remove_callback(self, callback: ResourceCallback) -> None:
        """Remove callback from execution list."""
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def start_real_time_monitoring(self, interval: float = 1.0) -> None:
        """Start real-time resource monitoring."""
        if self._monitoring_active:
            logger.warning("Resource monitoring is already active")
            return

        self._monitoring_active = True
        self._stop_event.clear()

        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            name="ResourceMonitor",
            daemon=True,
        )
        self._monitoring_thread.start()

        logger.info(
            f"Started real-time resource monitoring (interval={interval}s)"
        )

    def stop_monitoring(self) -> None:
        """Stop real-time resource monitoring."""
        if not self._monitoring_active:
            return

        self._monitoring_active = False
        self._stop_event.set()

        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5.0)

        logger.info("Stopped real-time resource monitoring")

    def get_current_snapshot(self) -> ResourceSnapshot:
        """Get current resource snapshot."""
        try:
            return self._collect_resource_snapshot()
        except Exception as e:
            raise MonitoringError(
                f"Failed to collect resource snapshot: {e}"
            ) from e

    def get_snapshots_history(
        self, count: int | None = None
    ) -> list[ResourceSnapshot]:
        """Get historical snapshots."""
        with self._lock:
            if count is None:
                return self._snapshots.copy()
            return self._snapshots[-count:]

    def clear_history(self) -> None:
        """Clear snapshot history."""
        with self._lock:
            self._snapshots.clear()

    def _monitoring_loop(self, interval: float) -> None:
        """Main monitoring loop for real-time tracking."""
        while not self._stop_event.wait(interval):
            try:
                snapshot = self._collect_resource_snapshot()

                # Store snapshot
                with self._lock:
                    self._snapshots.append(snapshot)
                    # Limit memory usage
                    if len(self._snapshots) > self._max_snapshots:
                        self._snapshots.pop(0)

                # Execute callbacks
                if self._callbacks:
                    resource_dict = snapshot.to_dict()
                    for callback in self._callbacks.copy():
                        try:
                            callback(resource_dict)
                        except Exception as e:
                            logger.error(
                                f"Error in resource monitoring callback: {e}"
                            )

                # Log to MonitoringManager if available
                if self.monitoring_manager:
                    manager_dict = self._snapshot_to_dict(snapshot)
                    self.monitoring_manager.log(manager_dict)

            except Exception as e:
                logger.error(f"Error in resource monitoring loop: {e}")

    def _collect_resource_snapshot(self) -> ResourceSnapshot:
        """Collect comprehensive resource snapshot."""
        timestamp = time.time()

        # System resources
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()

        # GPU resources
        gpu_info = self._get_gpu_info()

        # Process resources
        process_count = len(psutil.pids())
        thread_count = threading.active_count()
        file_handles = self._get_file_handles_count()

        # Network resources
        network_info = self._get_network_info()

        # Disk I/O
        disk_info = self._get_disk_io_info()

        # Application-specific
        temp_info = self._get_temp_files_info()

        return ResourceSnapshot(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            memory_used_mb=memory.used / (1024 * 1024),
            memory_available_mb=memory.available / (1024 * 1024),
            memory_percent=memory.percent,
            gpu_memory_used_mb=gpu_info["memory_used_mb"],
            gpu_memory_total_mb=gpu_info["memory_total_mb"],
            gpu_memory_percent=gpu_info["memory_percent"],
            gpu_utilization_percent=gpu_info["utilization_percent"],
            gpu_temperature_celsius=gpu_info["temperature_celsius"],
            process_count=process_count,
            thread_count=thread_count,
            file_handles=file_handles,
            network_connections=network_info["connections"],
            open_ports=network_info["open_ports"],
            disk_read_mb=disk_info["read_mb"],
            disk_write_mb=disk_info["write_mb"],
            temp_files_count=temp_info["count"],
            temp_files_size_mb=temp_info["size_mb"],
        )

    def _get_gpu_info(self) -> dict[str, float]:
        """Get GPU information (RTX 3070 Ti specific)."""
        if not self.enable_gpu_monitoring:
            return {
                "memory_used_mb": 0.0,
                "memory_total_mb": 0.0,
                "memory_percent": 0.0,
                "utilization_percent": 0.0,
                "temperature_celsius": 0.0,
            }

        try:
            import nvidia_ml_py3 as nml  # type: ignore[import-untyped]

            nml.nvmlInit()  # type: ignore[attr-defined]
            handle = nml.nvmlDeviceGetHandleByIndex(0)  # type: ignore[attr-defined]

            # Memory info
            memory_info = nml.nvmlDeviceGetMemoryInfo(handle)  # type: ignore[attr-defined]
            memory_used_mb = memory_info.used / (1024 * 1024)  # type: ignore[attr-defined]
            memory_total_mb = memory_info.total / (1024 * 1024)  # type: ignore[attr-defined]
            memory_percent = (memory_info.used / memory_info.total) * 100  # type: ignore[attr-defined]

            # Utilization
            util_info = nml.nvmlDeviceGetUtilizationRates(handle)  # type: ignore[attr-defined]
            utilization_percent = float(util_info.gpu)  # type: ignore[attr-defined]

            # Temperature
            temperature = nml.nvmlDeviceGetTemperature(  # type: ignore[attr-defined]
                handle,
                nml.NVML_TEMPERATURE_GPU,  # type: ignore[attr-defined]
            )

            return {
                "memory_used_mb": memory_used_mb,
                "memory_total_mb": memory_total_mb,
                "memory_percent": memory_percent,
                "utilization_percent": utilization_percent,
                "temperature_celsius": float(temperature),  # type: ignore[arg-type]
            }

        except (ImportError, Exception) as e:
            logger.debug(f"GPU monitoring unavailable: {e}")
            return {
                "memory_used_mb": 0.0,
                "memory_total_mb": 8192.0,  # RTX 3070 Ti default
                "memory_percent": 0.0,
                "utilization_percent": 0.0,
                "temperature_celsius": 0.0,
            }

    def _get_file_handles_count(self) -> int:
        """Get current file handles count."""
        if not self.enable_file_monitoring:
            return 0

        try:
            process = psutil.Process()
            return len(process.open_files())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0

    def _get_network_info(self) -> dict[str, Any]:
        """Get network connection information."""
        if not self.enable_network_monitoring:
            return {"connections": 0, "open_ports": []}

        try:
            connections = psutil.net_connections()
            open_ports = [
                conn.laddr.port for conn in connections if conn.laddr
            ]
            return {"connections": len(connections), "open_ports": open_ports}
        except (psutil.AccessDenied, AttributeError):
            return {"connections": 0, "open_ports": []}

    def _get_disk_io_info(self) -> dict[str, float]:
        """Get disk I/O information with delta calculation."""
        try:
            current_io = psutil.disk_io_counters()
            if not current_io or not self._disk_io_baseline:
                return {"read_mb": 0.0, "write_mb": 0.0}

            read_delta = (
                current_io.read_bytes - self._disk_io_baseline["read_bytes"]
            )
            write_delta = (
                current_io.write_bytes - self._disk_io_baseline["write_bytes"]
            )

            return {
                "read_mb": max(0, read_delta) / (1024 * 1024),
                "write_mb": max(0, write_delta) / (1024 * 1024),
            }
        except (AttributeError, OSError):
            return {"read_mb": 0.0, "write_mb": 0.0}

    def _get_temp_files_info(self) -> dict[str, Any]:
        """Get temporary files information."""
        temp_patterns = [
            Path("tools/utilities/temp_storage.py"),
            Path("generated_configs"),
            Path("artifacts/temp_*"),
            Path("test-artifacts/temp_*"),
            Path("selenium-videos/temp_*"),
        ]

        found_files: list[Path] = []
        total_size = 0.0

        for pattern in temp_patterns:
            if pattern.exists() and pattern.is_file():
                found_files.append(pattern)
                total_size += pattern.stat().st_size
            elif "*" in str(pattern):
                parent_dir = pattern.parent
                if parent_dir.exists():
                    pattern_name = pattern.name.replace("*", "")
                    for file_path in parent_dir.iterdir():
                        if (
                            pattern_name in file_path.name
                            and file_path.is_file()
                        ):
                            found_files.append(file_path)
                            total_size += file_path.stat().st_size

        return {
            "count": len(found_files),
            "size_mb": total_size / (1024 * 1024),
        }

    def _snapshot_to_dict(self, snapshot: ResourceSnapshot) -> ResourceDict:
        """Convert snapshot to dictionary for MonitoringManager."""
        return {
            "resource_monitor/cpu_percent": snapshot.cpu_percent,
            "resource_monitor/memory_used_mb": snapshot.memory_used_mb,
            "resource_monitor/memory_percent": snapshot.memory_percent,
            "resource_monitor/gpu_memory_used_mb": snapshot.gpu_memory_used_mb,
            "resource_monitor/gpu_memory_percent": snapshot.gpu_memory_percent,
            "resource_monitor/gpu_utilization_percent": (
                snapshot.gpu_utilization_percent
            ),
            "resource_monitor/gpu_temperature_celsius": (
                snapshot.gpu_temperature_celsius
            ),
            "resource_monitor/process_count": snapshot.process_count,
            "resource_monitor/thread_count": snapshot.thread_count,
            "resource_monitor/file_handles": snapshot.file_handles,
            "resource_monitor/network_connections": (
                snapshot.network_connections
            ),
            "resource_monitor/disk_read_mb": snapshot.disk_read_mb,
            "resource_monitor/disk_write_mb": snapshot.disk_write_mb,
            "resource_monitor/temp_files_count": snapshot.temp_files_count,
            "resource_monitor/temp_files_size_mb": snapshot.temp_files_size_mb,
        }
