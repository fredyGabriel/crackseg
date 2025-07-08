"""Threshold configuration for resource alerting.

This module provides threshold configuration for the alerting system,
with defaults optimized for crack segmentation workflows and RTX 3070 Ti.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ThresholdConfig:
    """Configuration for resource thresholds and alerting."""

    def __init__(self) -> None:
        """Initialize with crack segmentation and RTX 3070 Ti thresholds."""
        # CPU thresholds
        self.cpu_warning_percent = 80.0
        self.cpu_critical_percent = 90.0
        self.cpu_sustained_duration_s = 30.0

        # Memory thresholds
        self.memory_warning_percent = 75.0
        self.memory_critical_percent = 85.0
        self.memory_leak_growth_mb = 500.0

        # GPU thresholds (RTX 3070 Ti: 8GB VRAM)
        self.gpu_memory_warning_mb = 6000.0  # 75% of 8GB
        self.gpu_memory_critical_mb = 7000.0  # 87.5% of 8GB
        self.gpu_utilization_warning_percent = 85.0
        self.gpu_utilization_critical_percent = 95.0
        self.gpu_temperature_warning_celsius = 80.0
        self.gpu_temperature_critical_celsius = 85.0

        # Process thresholds
        self.max_process_count = 500
        self.max_file_handles = 1000
        self.max_thread_count = 100

        # Application thresholds
        self.temp_files_warning_mb = 1000.0
        self.temp_files_critical_mb = 2000.0
        self.max_network_connections = 100

        # Performance thresholds
        self.response_time_warning_ms = 2000.0
        self.response_time_critical_ms = 5000.0

    @classmethod
    def from_performance_thresholds(cls, thresholds: Any) -> "ThresholdConfig":
        """Create from PerformanceThresholds configuration.

        Args:
            thresholds: PerformanceThresholds instance from subtask 16.3

        Returns:
            ThresholdConfig instance with loaded values
        """
        config = cls()

        # Load from system_resources if available
        if hasattr(thresholds, "system_resources"):
            sys_res = thresholds.system_resources
            config.cpu_warning_percent = float(sys_res.cpu_warning_percent)
            config.cpu_critical_percent = float(sys_res.cpu_critical_percent)
            config.memory_warning_percent = (
                sys_res.memory_warning_mb / 1024.0 * 100.0
            )  # Convert to percentage approximation
            config.memory_critical_percent = (
                sys_res.memory_critical_mb / 1024.0 * 100.0
            )
            config.memory_leak_growth_mb = float(sys_res.memory_leak_growth_mb)
            logger.info("Loaded system resource thresholds from configuration")

        # Load from model_processing if available
        if hasattr(thresholds, "model_processing"):
            model_proc = thresholds.model_processing
            config.gpu_memory_warning_mb = float(model_proc.memory_warning_mb)
            config.gpu_memory_critical_mb = float(
                model_proc.memory_critical_mb
            )
            logger.info(
                "Loaded model processing thresholds from configuration"
            )

        # Load from network if available
        if hasattr(thresholds, "network"):
            network = thresholds.network
            if hasattr(network, "max_connections"):
                config.max_network_connections = int(network.max_connections)
            logger.info("Loaded network thresholds from configuration")

        return config

    def validate(self) -> bool:
        """Validate threshold configuration.

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration values are invalid
        """
        # Validate CPU thresholds
        if not (
            0 < self.cpu_warning_percent < self.cpu_critical_percent <= 100
        ):
            raise ValueError("Invalid CPU threshold percentages")

        # Validate memory thresholds
        if not (
            0
            < self.memory_warning_percent
            < self.memory_critical_percent
            <= 100
        ):
            raise ValueError("Invalid memory threshold percentages")

        # Validate GPU thresholds
        if self.gpu_memory_warning_mb >= self.gpu_memory_critical_mb:
            raise ValueError("GPU memory warning must be less than critical")

        if not (
            0
            < self.gpu_utilization_warning_percent
            < self.gpu_utilization_critical_percent
            <= 100
        ):
            raise ValueError("Invalid GPU utilization threshold percentages")

        # Validate temperature thresholds
        if (
            self.gpu_temperature_warning_celsius
            >= self.gpu_temperature_critical_celsius
        ):
            raise ValueError(
                "GPU temperature warning must be less than critical"
            )

        # Validate process thresholds
        if any(
            val <= 0
            for val in [
                self.max_process_count,
                self.max_file_handles,
                self.max_thread_count,
            ]
        ):
            raise ValueError("Process thresholds must be positive")

        logger.info("Threshold configuration validation passed")
        return True

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "cpu_warning_percent": self.cpu_warning_percent,
            "cpu_critical_percent": self.cpu_critical_percent,
            "cpu_sustained_duration_s": self.cpu_sustained_duration_s,
            "memory_warning_percent": self.memory_warning_percent,
            "memory_critical_percent": self.memory_critical_percent,
            "memory_leak_growth_mb": self.memory_leak_growth_mb,
            "gpu_memory_warning_mb": self.gpu_memory_warning_mb,
            "gpu_memory_critical_mb": self.gpu_memory_critical_mb,
            "gpu_utilization_warning_percent": (
                self.gpu_utilization_warning_percent
            ),
            "gpu_utilization_critical_percent": (
                self.gpu_utilization_critical_percent
            ),
            "gpu_temperature_warning_celsius": (
                self.gpu_temperature_warning_celsius
            ),
            "gpu_temperature_critical_celsius": (
                self.gpu_temperature_critical_celsius
            ),
            "max_process_count": self.max_process_count,
            "max_file_handles": self.max_file_handles,
            "max_thread_count": self.max_thread_count,
            "temp_files_warning_mb": self.temp_files_warning_mb,
            "temp_files_critical_mb": self.temp_files_critical_mb,
            "max_network_connections": self.max_network_connections,
            "response_time_warning_ms": self.response_time_warning_ms,
            "response_time_critical_ms": self.response_time_critical_ms,
        }
