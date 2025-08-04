"""Threshold checking logic for resource monitoring.

This module contains the logic for checking various resource thresholds
and generating appropriate alerts based on the configuration.
"""

import logging
import time

from ..resources.config import ThresholdConfig
from ..resources.snapshot import ResourceDict
from .types import Alert, AlertSeverity, AlertType

logger = logging.getLogger(__name__)


class ThresholdChecker:
    """Handles threshold checking logic for resource monitoring."""

    def __init__(self, threshold_config: ThresholdConfig) -> None:
        """Initialize threshold checker.

        Args:
            threshold_config: Configuration for threshold values
        """
        self.config = threshold_config

        # Track sustained violations for time-based thresholds
        self._cpu_high_start: float | None = None
        self._memory_baseline: float | None = None
        self._baseline_timestamp: float | None = None

    def check_all_thresholds(self, resource_dict: ResourceDict) -> list[Alert]:
        """Check all thresholds and return any alerts.

        Args:
            resource_dict: Current resource measurements

        Returns:
            List of alerts for threshold violations
        """
        alerts = []

        alerts.extend(self._check_cpu_thresholds(resource_dict))
        alerts.extend(self._check_memory_thresholds(resource_dict))
        alerts.extend(self._check_gpu_thresholds(resource_dict))
        alerts.extend(self._check_process_thresholds(resource_dict))
        alerts.extend(self._check_application_thresholds(resource_dict))

        return alerts

    def establish_baseline(self, resource_dict: ResourceDict) -> None:
        """Establish baseline values for leak detection."""
        self._memory_baseline = resource_dict.get("memory_used_mb", 0.0)
        self._baseline_timestamp = time.time()
        logger.info(
            f"Established memory baseline: {self._memory_baseline:.1f}MB"
        )

    def _check_cpu_thresholds(
        self, resource_dict: ResourceDict
    ) -> list[Alert]:
        """Check CPU usage thresholds."""
        alerts = []
        cpu_percent = resource_dict.get("cpu_percent", 0.0)
        current_time = time.time()

        # Critical CPU usage
        if cpu_percent >= self.config.cpu_critical_percent:
            alerts.append(
                self._create_alert(
                    AlertType.CPU_USAGE,
                    AlertSeverity.CRITICAL,
                    f"CPU usage critical: {cpu_percent:.1f}%",
                    cpu_percent,
                    self.config.cpu_critical_percent,
                    "cpu",
                )
            )
        # Warning CPU usage
        elif cpu_percent >= self.config.cpu_warning_percent:
            # Track sustained high CPU usage
            if self._cpu_high_start is None:
                self._cpu_high_start = current_time
            elif (
                current_time - self._cpu_high_start
            ) >= self.config.cpu_sustained_duration_s:
                alerts.append(
                    self._create_alert(
                        AlertType.CPU_USAGE,
                        AlertSeverity.WARNING,
                        f"Sustained high CPU usage: {cpu_percent:.1f}% "
                        f"for {current_time - self._cpu_high_start:.0f}s",
                        cpu_percent,
                        self.config.cpu_warning_percent,
                        "cpu",
                    )
                )
        else:
            # Reset sustained CPU tracking
            self._cpu_high_start = None

        return alerts

    def _check_memory_thresholds(
        self, resource_dict: ResourceDict
    ) -> list[Alert]:
        """Check memory usage and leak thresholds."""
        alerts = []
        memory_percent = resource_dict.get("memory_percent", 0.0)
        memory_used_mb = resource_dict.get("memory_used_mb", 0.0)

        # Critical memory usage
        if memory_percent >= self.config.memory_critical_percent:
            alerts.append(
                self._create_alert(
                    AlertType.MEMORY_USAGE,
                    AlertSeverity.CRITICAL,
                    f"Memory usage critical: {memory_percent:.1f}%",
                    memory_percent,
                    self.config.memory_critical_percent,
                    "memory",
                )
            )
        # Warning memory usage
        elif memory_percent >= self.config.memory_warning_percent:
            alerts.append(
                self._create_alert(
                    AlertType.MEMORY_USAGE,
                    AlertSeverity.WARNING,
                    f"Memory usage high: {memory_percent:.1f}%",
                    memory_percent,
                    self.config.memory_warning_percent,
                    "memory",
                )
            )

        # Memory leak detection
        if (
            self._memory_baseline is not None
            and self._baseline_timestamp is not None
        ):
            memory_growth = memory_used_mb - self._memory_baseline
            if memory_growth >= self.config.memory_leak_growth_mb:
                time_elapsed = time.time() - self._baseline_timestamp
                alerts.append(
                    self._create_alert(
                        AlertType.MEMORY_LEAK,
                        AlertSeverity.WARNING,
                        f"Potential memory leak: {memory_growth:.1f}MB "
                        f"growth in {time_elapsed:.0f}s",
                        memory_growth,
                        self.config.memory_leak_growth_mb,
                        "memory_leak",
                    )
                )

        return alerts

    def _check_gpu_thresholds(
        self, resource_dict: ResourceDict
    ) -> list[Alert]:
        """Check GPU resource thresholds (RTX 3070 Ti specific)."""
        alerts = []

        gpu_memory_mb = resource_dict.get("gpu_memory_used_mb", 0.0)
        gpu_utilization = resource_dict.get("gpu_utilization_percent", 0.0)
        gpu_temperature = resource_dict.get("gpu_temperature_celsius", 0.0)

        # GPU memory thresholds
        if gpu_memory_mb >= self.config.gpu_memory_critical_mb:
            alerts.append(
                self._create_alert(
                    AlertType.GPU_MEMORY_USAGE,
                    AlertSeverity.CRITICAL,
                    f"GPU memory critical: {gpu_memory_mb:.0f}MB "
                    f"(RTX 3070 Ti limit approaching)",
                    gpu_memory_mb,
                    self.config.gpu_memory_critical_mb,
                    "gpu_memory",
                )
            )
        elif gpu_memory_mb >= self.config.gpu_memory_warning_mb:
            alerts.append(
                self._create_alert(
                    AlertType.GPU_MEMORY_USAGE,
                    AlertSeverity.WARNING,
                    f"GPU memory high: {gpu_memory_mb:.0f}MB",
                    gpu_memory_mb,
                    self.config.gpu_memory_warning_mb,
                    "gpu_memory",
                )
            )

        # GPU utilization thresholds
        if gpu_utilization >= self.config.gpu_utilization_critical_percent:
            alerts.append(
                self._create_alert(
                    AlertType.GPU_UTILIZATION,
                    AlertSeverity.CRITICAL,
                    f"GPU utilization critical: {gpu_utilization:.1f}%",
                    gpu_utilization,
                    self.config.gpu_utilization_critical_percent,
                    "gpu_utilization",
                )
            )
        elif gpu_utilization >= self.config.gpu_utilization_warning_percent:
            alerts.append(
                self._create_alert(
                    AlertType.GPU_UTILIZATION,
                    AlertSeverity.WARNING,
                    f"GPU utilization high: {gpu_utilization:.1f}%",
                    gpu_utilization,
                    self.config.gpu_utilization_warning_percent,
                    "gpu_utilization",
                )
            )

        # GPU temperature thresholds
        if gpu_temperature >= self.config.gpu_temperature_critical_celsius:
            alerts.append(
                self._create_alert(
                    AlertType.GPU_TEMPERATURE,
                    AlertSeverity.CRITICAL,
                    f"GPU temperature critical: {gpu_temperature:.1f}°C",
                    gpu_temperature,
                    self.config.gpu_temperature_critical_celsius,
                    "gpu_temperature",
                )
            )
        elif gpu_temperature >= self.config.gpu_temperature_warning_celsius:
            alerts.append(
                self._create_alert(
                    AlertType.GPU_TEMPERATURE,
                    AlertSeverity.WARNING,
                    f"GPU temperature high: {gpu_temperature:.1f}°C",
                    gpu_temperature,
                    self.config.gpu_temperature_warning_celsius,
                    "gpu_temperature",
                )
            )

        return alerts

    def _check_process_thresholds(
        self, resource_dict: ResourceDict
    ) -> list[Alert]:
        """Check process-related thresholds."""
        alerts = []

        process_count = resource_dict.get("process_count", 0)
        file_handles = resource_dict.get("file_handles", 0)
        thread_count = resource_dict.get("thread_count", 0)

        # Process count threshold
        if process_count >= self.config.max_process_count:
            alerts.append(
                self._create_alert(
                    AlertType.PROCESS_COUNT,
                    AlertSeverity.WARNING,
                    f"High process count: {process_count}",
                    float(process_count),
                    float(self.config.max_process_count),
                    "processes",
                )
            )

        # File handles threshold
        if file_handles >= self.config.max_file_handles:
            alerts.append(
                self._create_alert(
                    AlertType.FILE_HANDLES,
                    AlertSeverity.WARNING,
                    f"High file handle count: {file_handles}",
                    float(file_handles),
                    float(self.config.max_file_handles),
                    "file_handles",
                )
            )

        # Thread count threshold
        if thread_count >= self.config.max_thread_count:
            alerts.append(
                self._create_alert(
                    AlertType.THREAD_COUNT,
                    AlertSeverity.WARNING,
                    f"High thread count: {thread_count}",
                    float(thread_count),
                    float(self.config.max_thread_count),
                    "threads",
                )
            )

        return alerts

    def _check_application_thresholds(
        self, resource_dict: ResourceDict
    ) -> list[Alert]:
        """Check application-specific thresholds."""
        alerts = []

        temp_files_mb = resource_dict.get("temp_files_size_mb", 0.0)
        network_connections = resource_dict.get("network_connections", 0)

        # Temporary files accumulation
        if temp_files_mb >= self.config.temp_files_critical_mb:
            alerts.append(
                self._create_alert(
                    AlertType.TEMP_FILES_ACCUMULATION,
                    AlertSeverity.CRITICAL,
                    f"Excessive temporary files: {temp_files_mb:.1f}MB",
                    temp_files_mb,
                    self.config.temp_files_critical_mb,
                    "temp_files",
                )
            )
        elif temp_files_mb >= self.config.temp_files_warning_mb:
            alerts.append(
                self._create_alert(
                    AlertType.TEMP_FILES_ACCUMULATION,
                    AlertSeverity.WARNING,
                    f"Growing temporary files: {temp_files_mb:.1f}MB",
                    temp_files_mb,
                    self.config.temp_files_warning_mb,
                    "temp_files",
                )
            )

        # Network connections threshold
        if network_connections >= self.config.max_network_connections:
            alerts.append(
                self._create_alert(
                    AlertType.NETWORK_CONNECTIONS,
                    AlertSeverity.WARNING,
                    f"High network connections: {network_connections}",
                    float(network_connections),
                    float(self.config.max_network_connections),
                    "network",
                )
            )

        return alerts

    def _create_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        message: str,
        value: float,
        threshold: float,
        resource_name: str = "",
    ) -> Alert:
        """Create an alert with context information."""
        return Alert(
            alert_type=alert_type,
            severity=severity,
            message=message,
            value=value,
            threshold=threshold,
            resource_name=resource_name,
            context={
                "checker": "ThresholdChecker",
                "config_values": self.config.to_dict(),
            },
        )
