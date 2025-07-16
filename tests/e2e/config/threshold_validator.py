"""Threshold validation system for performance monitoring.

This module provides integration between the MonitoringManager and
performance thresholds, enabling real-time validation of metrics
against configured SLA boundaries.

Example:
    >>> from crackseg.utils.monitoring.manager import MonitoringManager
    >>> from tests.e2e.config.threshold_validator import ThresholdValidator
    >>>
    >>> monitoring = MonitoringManager()
    >>> validator = ThresholdValidator.from_config_file(
    ...     "configs/testing/performance_thresholds.yaml"
    ... )
    >>>
    >>> # Log metrics and validate
    >>> monitoring.log({"page_load_time_ms": 1800})
    >>> violations = validator.validate_metrics(monitoring)
    >>> if violations:
    >>>     print(f"SLA violations detected: {violations}")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir

from crackseg.utils.core.exceptions import ValidationError

from .performance_thresholds import PerformanceThresholds

logger = logging.getLogger(__name__)


class ViolationSeverity(Enum):
    """Severity levels for threshold violations."""

    WARNING = "warning"
    CRITICAL = "critical"
    TIMEOUT = "timeout"


@dataclass
class ThresholdViolation:
    """Represents a single threshold violation."""

    metric_name: str
    actual_value: float | int
    threshold_value: float | int
    severity: ViolationSeverity
    timestamp: float
    context: str
    message: str

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"{self.severity.value.upper()}: {self.metric_name} = "
            f"{self.actual_value} exceeds {self.threshold_value} threshold "
            f"in context '{self.context}'"
        )


class ThresholdValidator:
    """Validates metrics against performance thresholds.

    Integrates with MonitoringManager to provide real-time
    threshold violation detection and reporting.
    """

    def __init__(self, thresholds: PerformanceThresholds) -> None:
        """Initialize with performance thresholds.

        Args:
            thresholds: Validated performance thresholds configuration.
        """
        self.thresholds = thresholds
        self.violations_history: list[ThresholdViolation] = []
        self._last_validation_time: float = 0.0

    @classmethod
    def from_config_file(cls, config_path: str | Path) -> ThresholdValidator:
        """Create validator from Hydra configuration file.

        Args:
            config_path: Path to performance thresholds YAML file.

        Returns:
            Configured ThresholdValidator instance.

        Raises:
            ValidationError: If configuration loading or validation fails.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise ValidationError(
                f"Configuration file not found: {config_path}"
            )

        try:
            # Use Hydra to load configuration
            config_dir = str(config_path.parent.absolute())
            config_name = config_path.stem

            with initialize_config_dir(
                config_dir=config_dir, version_base=None
            ):
                cfg = compose(config_name=config_name)
                thresholds = PerformanceThresholds.from_config(cfg)
                thresholds.validate()

            logger.info(f"Loaded performance thresholds from {config_path}")
            return cls(thresholds)

        except Exception as e:
            raise ValidationError(
                f"Failed to load threshold configuration: {e}"
            ) from e

    def validate_metrics(
        self,
        monitoring_manager: Any,  # MonitoringManager from monitoring package
        context_filter: str | None = None,
    ) -> list[ThresholdViolation]:
        """Validate current metrics against thresholds.

        Args:
            monitoring_manager: MonitoringManager instance with collected
                metrics.
            context_filter: Optional context filter (e.g., 'train', 'val',
               'test').

        Returns:
            List of threshold violations detected.
        """
        violations = []
        current_time = time.time()
        self._last_validation_time = current_time

        # Get metrics history from monitoring manager
        metrics_history = monitoring_manager.get_history()

        # Web interface metrics validation
        violations.extend(
            self._validate_web_interface_metrics(
                metrics_history, current_time, context_filter
            )
        )

        # Model processing metrics validation
        violations.extend(
            self._validate_model_processing_metrics(
                metrics_history, current_time, context_filter
            )
        )

        # System resource metrics validation
        violations.extend(
            self._validate_system_resource_metrics(
                metrics_history, current_time, context_filter
            )
        )

        # Container management metrics validation
        violations.extend(
            self._validate_container_metrics(
                metrics_history, current_time, context_filter
            )
        )

        # File operation metrics validation
        violations.extend(
            self._validate_file_operation_metrics(
                metrics_history, current_time, context_filter
            )
        )

        # Store violations in history
        self.violations_history.extend(violations)

        if violations:
            logger.warning(f"Detected {len(violations)} threshold violations")
            for violation in violations:
                logger.warning(str(violation))

        return violations

    def _validate_web_interface_metrics(
        self,
        metrics: dict[str, list[Any]],
        timestamp: float,
        context_filter: str | None,
    ) -> list[ThresholdViolation]:
        """Validate web interface performance metrics."""
        violations = []

        # Page load time validation
        page_load_values = self._get_latest_values(
            metrics, "page_load_time_ms", context_filter
        )
        for value, context in page_load_values:
            violations.extend(
                self._check_threshold_levels(
                    metric_name="page_load_time_ms",
                    value=value,
                    warning_threshold=self.thresholds.web_interface.page_load_warning_ms,
                    critical_threshold=self.thresholds.web_interface.page_load_critical_ms,
                    context=context,
                    timestamp=timestamp,
                )
            )

        # Config validation time validation
        config_validation_values = self._get_latest_values(
            metrics, "config_validation_time_ms", context_filter
        )
        for value, context in config_validation_values:
            violations.extend(
                self._check_threshold_levels(
                    metric_name="config_validation_time_ms",
                    value=value,
                    warning_threshold=self.thresholds.web_interface.config_validation_warning_ms,
                    critical_threshold=self.thresholds.web_interface.config_validation_critical_ms,
                    context=context,
                    timestamp=timestamp,
                )
            )

        return violations

    def _validate_model_processing_metrics(
        self,
        metrics: dict[str, list[Any]],
        timestamp: float,
        context_filter: str | None,
    ) -> list[ThresholdViolation]:
        """Validate model processing performance metrics."""
        violations = []

        # Inference time validation
        inference_values = self._get_latest_values(
            metrics, "inference_time_ms", context_filter
        )
        for value, context in inference_values:
            violations.extend(
                self._check_threshold_levels(
                    metric_name="inference_time_ms",
                    value=value,
                    warning_threshold=self.thresholds.model_processing.inference_warning_ms,
                    critical_threshold=self.thresholds.model_processing.inference_critical_ms,
                    context=context,
                    timestamp=timestamp,
                )
            )

        # Memory usage validation
        memory_values = self._get_latest_values(
            metrics, "gpu_memory_used_mb", context_filter
        )
        for value, context in memory_values:
            violations.extend(
                self._check_threshold_levels(
                    metric_name="gpu_memory_used_mb",
                    value=value,
                    warning_threshold=self.thresholds.model_processing.memory_warning_mb,
                    critical_threshold=self.thresholds.model_processing.memory_critical_mb,
                    context=context,
                    timestamp=timestamp,
                )
            )

        return violations

    def _validate_system_resource_metrics(
        self,
        metrics: dict[str, list[Any]],
        timestamp: float,
        context_filter: str | None,
    ) -> list[ThresholdViolation]:
        """Validate system resource consumption metrics."""
        violations = []

        # CPU usage validation
        cpu_values = self._get_latest_values(
            metrics, "cpu_usage_percent", context_filter
        )
        for value, context in cpu_values:
            violations.extend(
                self._check_threshold_levels(
                    metric_name="cpu_usage_percent",
                    value=value,
                    warning_threshold=self.thresholds.system_resources.cpu_warning_percent,
                    critical_threshold=self.thresholds.system_resources.cpu_critical_percent,
                    context=context,
                    timestamp=timestamp,
                )
            )

        # Memory usage validation
        memory_values = self._get_latest_values(
            metrics, "memory_usage_mb", context_filter
        )
        for value, context in memory_values:
            violations.extend(
                self._check_threshold_levels(
                    metric_name="memory_usage_mb",
                    value=value,
                    warning_threshold=self.thresholds.system_resources.memory_warning_mb,
                    critical_threshold=self.thresholds.system_resources.memory_critical_mb,
                    context=context,
                    timestamp=timestamp,
                )
            )

        return violations

    def _validate_container_metrics(
        self,
        metrics: dict[str, list[Any]],
        timestamp: float,
        context_filter: str | None,
    ) -> list[ThresholdViolation]:
        """Validate container management metrics."""
        violations = []

        # Container startup time validation
        startup_values = self._get_latest_values(
            metrics, "container_startup_time_s", context_filter
        )
        for value, context in startup_values:
            violations.extend(
                self._check_threshold_levels(
                    metric_name="container_startup_time_s",
                    value=value,
                    warning_threshold=self.thresholds.container_management.startup_warning_s,
                    critical_threshold=self.thresholds.container_management.startup_critical_s,
                    context=context,
                    timestamp=timestamp,
                )
            )

        return violations

    def _validate_file_operation_metrics(
        self,
        metrics: dict[str, list[Any]],
        timestamp: float,
        context_filter: str | None,
    ) -> list[ThresholdViolation]:
        """Validate file I/O operation metrics."""
        violations = []

        # File read time validation
        read_values = self._get_latest_values(
            metrics, "file_read_time_ms", context_filter
        )
        for value, context in read_values:
            violations.extend(
                self._check_threshold_levels(
                    metric_name="file_read_time_ms",
                    value=value,
                    warning_threshold=self.thresholds.file_operations.read_warning_ms,
                    critical_threshold=self.thresholds.file_operations.read_critical_ms,
                    context=context,
                    timestamp=timestamp,
                )
            )

        return violations

    def _get_latest_values(
        self,
        metrics: dict[str, list[Any]],
        metric_name: str,
        context_filter: str | None,
    ) -> list[tuple[float | int, str]]:
        """Extract latest values for a specific metric with context."""
        values = []

        for key, value_list in metrics.items():
            if f"/{metric_name}_values" in key and value_list:
                # Extract context from key
                # (e.g., "train/metric_name_values" -> "train")
                context = key.split("/")[0]

                if context_filter is None or context == context_filter:
                    latest_value = value_list[-1]  # Get most recent value
                    values.append((latest_value, context))

        return values

    def _check_threshold_levels(
        self,
        metric_name: str,
        value: float | int,
        warning_threshold: float | int,
        critical_threshold: float | int,
        context: str,
        timestamp: float,
    ) -> list[ThresholdViolation]:
        """Check value against warning and critical thresholds."""
        violations = []

        if value >= critical_threshold:
            violations.append(
                ThresholdViolation(
                    metric_name=metric_name,
                    actual_value=value,
                    threshold_value=critical_threshold,
                    severity=ViolationSeverity.CRITICAL,
                    timestamp=timestamp,
                    context=context,
                    message=(
                        f"Critical threshold exceeded: {value} >= "
                        f"{critical_threshold}"
                    ),
                )
            )
        elif value >= warning_threshold:
            violations.append(
                ThresholdViolation(
                    metric_name=metric_name,
                    actual_value=value,
                    threshold_value=warning_threshold,
                    severity=ViolationSeverity.WARNING,
                    timestamp=timestamp,
                    context=context,
                    message=(
                        f"Warning threshold exceeded: {value} >= "
                        f"{warning_threshold}"
                    ),
                )
            )

        return violations

    def get_violations_summary(self) -> dict[str, Any]:
        """Get summary of all detected violations.

        Returns:
            Summary statistics of threshold violations.
        """
        total_violations = len(self.violations_history)
        if total_violations == 0:
            return {"total_violations": 0, "status": "healthy"}

        severity_counts = {
            severity.value: sum(
                1 for v in self.violations_history if v.severity == severity
            )
            for severity in ViolationSeverity
        }

        recent_violations = [
            v
            for v in self.violations_history
            if v.timestamp > (time.time() - 3600)  # Last hour
        ]

        return {
            "total_violations": total_violations,
            "severity_breakdown": severity_counts,
            "recent_violations_count": len(recent_violations),
            "last_validation_time": self._last_validation_time,
            "status": "degraded" if recent_violations else "stable",
        }

    def clear_violations_history(self) -> None:
        """Clear violations history for fresh monitoring session."""
        self.violations_history.clear()
        logger.info("Cleared threshold violations history")
