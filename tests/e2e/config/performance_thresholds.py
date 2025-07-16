"""Type-safe performance threshold models for E2E testing pipeline.

This module provides dataclass models for validating performance thresholds
loaded from Hydra configuration. Ensures type safety while maintaining
operational flexibility through configuration files.

Example:
    >>> from hydra import compose, initialize
    >>> from tests.e2e.config.performance_thresholds import (
    ...     PerformanceThresholds,
    ... )
    >>>
    >>> with initialize(config_path="../../../configs"):
    >>>     cfg = compose(config_name="testing/performance_thresholds")
    >>>     thresholds = PerformanceThresholds.from_config(cfg)
    >>>     thresholds.validate()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from crackseg.utils.core.exceptions import ValidationError

logger = logging.getLogger(__name__)


@dataclass
class WebInterfaceThresholds:
    """Performance thresholds for web interface operations."""

    # Page load performance
    page_load_warning_ms: int
    page_load_critical_ms: int
    page_load_timeout_ms: int

    # Configuration validation performance
    config_validation_warning_ms: int
    config_validation_critical_ms: int
    config_validation_timeout_ms: int

    # User interaction responsiveness
    button_response_ms: int
    form_submission_ms: int
    file_upload_timeout_ms: int

    def __post_init__(self) -> None:
        """Validate web interface threshold consistency."""
        if self.page_load_warning_ms >= self.page_load_critical_ms:
            raise ValidationError(
                "Page load warning threshold must be less than critical "
                "threshold"
            )

        if self.page_load_critical_ms >= self.page_load_timeout_ms:
            raise ValidationError(
                "Page load critical threshold must be less than timeout"
            )

        if (
            self.config_validation_warning_ms
            >= self.config_validation_critical_ms
        ):
            raise ValidationError(
                "Config validation warning must be less than critical "
                "threshold"
            )

    @classmethod
    def from_config(cls, config: Any) -> WebInterfaceThresholds:
        """Create from Hydra configuration object."""
        web_config = config.web_interface
        return cls(
            page_load_warning_ms=web_config.page_load_time.warning_threshold_ms,
            page_load_critical_ms=web_config.page_load_time.critical_threshold_ms,
            page_load_timeout_ms=web_config.page_load_time.timeout_ms,
            config_validation_warning_ms=web_config.config_validation.warning_threshold_ms,
            config_validation_critical_ms=web_config.config_validation.critical_threshold_ms,
            config_validation_timeout_ms=web_config.config_validation.timeout_ms,
            button_response_ms=web_config.user_interaction.button_response_ms,
            form_submission_ms=web_config.user_interaction.form_submission_ms,
            file_upload_timeout_ms=web_config.user_interaction.file_upload_timeout_ms,
        )


@dataclass
class ModelProcessingThresholds:
    """Performance thresholds for model processing operations."""

    # Inference timing
    inference_warning_ms: int
    inference_critical_ms: int
    batch_timeout_ms: int

    # Memory usage (RTX 3070 Ti specific)
    memory_warning_mb: int
    memory_critical_mb: int
    oom_threshold_mb: int

    # Quality metrics
    min_precision: float
    min_recall: float
    min_iou: float

    def __post_init__(self) -> None:
        """Validate model processing threshold consistency."""
        if self.inference_warning_ms >= self.inference_critical_ms:
            raise ValidationError(
                "Inference warning threshold must be less than critical "
                "threshold"
            )

        if self.memory_warning_mb >= self.memory_critical_mb:
            raise ValidationError(
                "Memory warning threshold must be less than critical threshold"
            )

        if self.memory_critical_mb >= self.oom_threshold_mb:
            raise ValidationError(
                "Memory critical threshold must be less than OOM threshold"
            )

        # Validate quality metrics are in valid ranges
        for metric_name, metric_value in [
            ("precision", self.min_precision),
            ("recall", self.min_recall),
            ("iou", self.min_iou),
        ]:
            if not 0.0 <= metric_value <= 1.0:
                raise ValidationError(
                    f"Minimum {metric_name} must be between 0.0 and 1.0, "
                    f"got {metric_value}"
                )

    @classmethod
    def from_config(cls, config: Any) -> ModelProcessingThresholds:
        """Create from Hydra configuration object."""
        model_config = config.model_processing
        return cls(
            inference_warning_ms=model_config.inference_time.warning_threshold_ms,
            inference_critical_ms=model_config.inference_time.critical_threshold_ms,
            batch_timeout_ms=model_config.inference_time.batch_timeout_ms,
            memory_warning_mb=model_config.memory_usage.warning_threshold_mb,
            memory_critical_mb=model_config.memory_usage.critical_threshold_mb,
            oom_threshold_mb=model_config.memory_usage.oom_threshold_mb,
            min_precision=model_config.crack_detection.min_precision,
            min_recall=model_config.crack_detection.min_recall,
            min_iou=model_config.crack_detection.min_iou,
        )


@dataclass
class SystemResourceThresholds:
    """Performance thresholds for system resource consumption."""

    # CPU usage
    cpu_warning_percent: int
    cpu_critical_percent: int
    cpu_sustained_duration_s: int

    # Memory usage
    memory_warning_mb: int
    memory_critical_mb: int
    memory_leak_growth_mb: int

    # Disk usage
    temp_files_warning_mb: int
    temp_files_critical_mb: int
    log_files_max_mb: int

    def __post_init__(self) -> None:
        """Validate system resource threshold consistency."""
        if self.cpu_warning_percent >= self.cpu_critical_percent:
            raise ValidationError(
                "CPU warning threshold must be less than critical threshold"
            )

        if self.memory_warning_mb >= self.memory_critical_mb:
            raise ValidationError(
                "Memory warning threshold must be less than critical threshold"
            )

        if self.temp_files_warning_mb >= self.temp_files_critical_mb:
            raise ValidationError(
                "Temp files warning threshold must be less than critical "
                "threshold"
            )

        # Validate percentage ranges
        for percent_name, percent_value in [
            ("cpu_warning", self.cpu_warning_percent),
            ("cpu_critical", self.cpu_critical_percent),
        ]:
            if not 0 <= percent_value <= 100:
                raise ValidationError(
                    f"{percent_name} percentage must be between 0 and 100, "
                    f"got {percent_value}"
                )

    @classmethod
    def from_config(cls, config: Any) -> SystemResourceThresholds:
        """Create from Hydra configuration object."""
        sys_config = config.system_resources
        return cls(
            cpu_warning_percent=sys_config.cpu_usage.warning_threshold_percent,
            cpu_critical_percent=sys_config.cpu_usage.critical_threshold_percent,
            cpu_sustained_duration_s=sys_config.cpu_usage.sustained_duration_s,
            memory_warning_mb=sys_config.memory_usage.warning_threshold_mb,
            memory_critical_mb=sys_config.memory_usage.critical_threshold_mb,
            memory_leak_growth_mb=sys_config.memory_usage.leak_detection_growth_mb,
            temp_files_warning_mb=sys_config.disk_usage.temp_files_warning_mb,
            temp_files_critical_mb=sys_config.disk_usage.temp_files_critical_mb,
            log_files_max_mb=sys_config.disk_usage.log_files_max_mb,
        )


@dataclass
class ContainerManagementThresholds:
    """Performance thresholds for Docker container operations."""

    # Startup timing
    startup_warning_s: int
    startup_critical_s: int
    startup_timeout_s: int

    # Shutdown timing
    shutdown_warning_s: int
    shutdown_critical_s: int
    force_kill_timeout_s: int

    # Resource cleanup limits
    max_orphaned_containers: int
    max_dangling_images: int
    max_unused_volumes: int

    def __post_init__(self) -> None:
        """Validate container management threshold consistency."""
        if self.startup_warning_s >= self.startup_critical_s:
            raise ValidationError(
                "Container startup warning must be less than critical "
                "threshold"
            )

        if self.startup_critical_s >= self.startup_timeout_s:
            raise ValidationError(
                "Container startup critical must be less than timeout"
            )

        if self.shutdown_warning_s >= self.shutdown_critical_s:
            raise ValidationError(
                "Container shutdown warning must be less than critical "
                "threshold"
            )

    @classmethod
    def from_config(cls, config: Any) -> ContainerManagementThresholds:
        """Create from Hydra configuration object."""
        container_config = config.container_management
        return cls(
            startup_warning_s=container_config.startup_time.warning_threshold_s,
            startup_critical_s=container_config.startup_time.critical_threshold_s,
            startup_timeout_s=container_config.startup_time.timeout_s,
            shutdown_warning_s=container_config.shutdown_time.warning_threshold_s,
            shutdown_critical_s=container_config.shutdown_time.critical_threshold_s,
            force_kill_timeout_s=container_config.shutdown_time.force_kill_timeout_s,
            max_orphaned_containers=container_config.resource_cleanup.orphaned_containers_max,
            max_dangling_images=container_config.resource_cleanup.dangling_images_max,
            max_unused_volumes=container_config.resource_cleanup.unused_volumes_max,
        )


@dataclass
class FileOperationThresholds:
    """Performance thresholds for file I/O operations."""

    # Test data access timing
    read_warning_ms: int
    read_critical_ms: int
    write_warning_ms: int
    write_critical_ms: int

    # Artifact generation timing
    screenshot_save_ms: int
    report_generation_ms: int
    cleanup_completion_ms: int

    # Concurrent access limits
    max_open_files: int
    lock_timeout_ms: int

    def __post_init__(self) -> None:
        """Validate file operation threshold consistency."""
        if self.read_warning_ms >= self.read_critical_ms:
            raise ValidationError(
                "File read warning threshold must be less than critical "
                "threshold"
            )

        if self.write_warning_ms >= self.write_critical_ms:
            raise ValidationError(
                "File write warning threshold must be less than critical "
                "threshold"
            )

        if self.max_open_files <= 0:
            raise ValidationError("Maximum open files must be positive")

    @classmethod
    def from_config(cls, config: Any) -> FileOperationThresholds:
        """Create from Hydra configuration object."""
        file_config = config.file_operations
        return cls(
            read_warning_ms=file_config.test_data_access.read_time_warning_ms,
            read_critical_ms=file_config.test_data_access.read_time_critical_ms,
            write_warning_ms=file_config.test_data_access.write_time_warning_ms,
            write_critical_ms=file_config.test_data_access.write_time_critical_ms,
            screenshot_save_ms=file_config.artifact_generation.screenshot_save_ms,
            report_generation_ms=file_config.artifact_generation.report_generation_ms,
            cleanup_completion_ms=file_config.artifact_generation.cleanup_completion_ms,
            max_open_files=file_config.concurrent_access.max_open_files,
            lock_timeout_ms=file_config.concurrent_access.lock_timeout_ms,
        )


@dataclass
class PerformanceThresholds:
    """Master container for all performance thresholds.

    This class aggregates all threshold categories and provides
    unified validation and configuration loading.
    """

    web_interface: WebInterfaceThresholds
    model_processing: ModelProcessingThresholds
    system_resources: SystemResourceThresholds
    container_management: ContainerManagementThresholds
    file_operations: FileOperationThresholds

    def validate(self) -> None:
        """Validate all threshold configurations.

        Raises:
            ValidationError: If any threshold configuration is invalid.
        """
        logger.info("Validating performance thresholds configuration")

        # Individual threshold validation happens in __post_init__
        # This method can perform cross-category validation if needed

        logger.info("Performance thresholds validation completed successfully")

    @classmethod
    def from_config(cls, config: Any) -> PerformanceThresholds:
        """Create from Hydra configuration object.

        Args:
            config: Hydra configuration object loaded from YAML.

        Returns:
            Validated PerformanceThresholds instance.

        Raises:
            ValidationError: If configuration is invalid.
        """
        try:
            return cls(
                web_interface=WebInterfaceThresholds.from_config(config),
                model_processing=ModelProcessingThresholds.from_config(config),
                system_resources=SystemResourceThresholds.from_config(config),
                container_management=ContainerManagementThresholds.from_config(
                    config
                ),
                file_operations=FileOperationThresholds.from_config(config),
            )
        except Exception as e:
            raise ValidationError(
                f"Failed to load performance thresholds: {e}"
            ) from e

    def get_summary(self) -> dict[str, str]:
        """Get human-readable summary of key thresholds.

        Returns:
            Dictionary with key threshold values for logging/reporting.
        """
        return {
            "page_load_sla_ms": str(self.web_interface.page_load_critical_ms),
            "config_validation_sla_ms": str(
                self.web_interface.config_validation_critical_ms
            ),
            "inference_sla_ms": str(
                self.model_processing.inference_critical_ms
            ),
            "vram_limit_mb": str(self.model_processing.memory_critical_mb),
            "cpu_limit_percent": str(
                self.system_resources.cpu_critical_percent
            ),
            "container_startup_sla_s": str(
                self.container_management.startup_critical_s
            ),
        }
