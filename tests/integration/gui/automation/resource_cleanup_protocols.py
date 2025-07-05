"""Resource cleanup validation protocols and data structures.

This module provides the core data structures and protocols for resource
cleanup validation, including metrics, baselines, and interfaces for systematic
resource management verification across workflow components (9.1-9.4).
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol


@dataclass
class ResourceCleanupMetrics:
    """Comprehensive metrics for resource cleanup validation."""

    # Memory cleanup metrics
    memory_baseline_mb: float
    memory_post_execution_mb: float
    memory_leak_detected: bool
    memory_cleanup_percentage: float

    # Process cleanup metrics
    process_baseline_count: int
    process_post_execution_count: int
    orphaned_processes_detected: bool
    process_cleanup_successful: bool

    # File system cleanup metrics
    file_handles_baseline: int
    file_handles_post_execution: int
    file_leak_detected: bool
    temp_files_cleaned: bool

    # GPU resource cleanup (RTX 3070 Ti specific)
    gpu_memory_baseline_mb: float
    gpu_memory_post_execution_mb: float
    gpu_memory_leaked: bool
    cuda_context_cleaned: bool

    # System resource restoration
    cpu_baseline_percent: float
    cpu_post_execution_percent: float
    baseline_restoration_successful: bool

    # Cleanup validation results
    cleanup_validation_passed: bool
    cleanup_time_seconds: float
    leak_detection_accuracy: float

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    workflow_component: str = ""
    cleanup_phase: str = ""

    def get_memory_leak_severity(self) -> str:
        """Get memory leak severity level."""
        if not self.memory_leak_detected:
            return "none"

        leak_percentage = (
            (self.memory_post_execution_mb - self.memory_baseline_mb)
            / self.memory_baseline_mb
            * 100
        )

        if leak_percentage > 20:
            return "critical"
        elif leak_percentage > 10:
            return "major"
        elif leak_percentage > 5:
            return "minor"
        else:
            return "negligible"

    def get_cleanup_summary(self) -> dict[str, Any]:
        """Get summary of cleanup validation results."""
        return {
            "overall_success": self.cleanup_validation_passed,
            "memory_leak_severity": self.get_memory_leak_severity(),
            "processes_cleaned": self.process_cleanup_successful,
            "baseline_restored": self.baseline_restoration_successful,
            "cleanup_duration": self.cleanup_time_seconds,
        }


@dataclass
class ResourceBaseline:
    """Baseline resource state for comparison after cleanup."""

    memory_mb: float
    process_count: int
    file_handles: int
    gpu_memory_mb: float
    cpu_percent: float
    open_ports: list[int] = field(default_factory=list)
    temp_files: list[Path] = field(default_factory=list)
    thread_count: int = 0
    established_time: datetime = field(default_factory=datetime.now)

    def get_baseline_age_minutes(self) -> float:
        """Get age of baseline in minutes."""
        age = datetime.now() - self.established_time
        return age.total_seconds() / 60.0

    def is_baseline_stale(self, max_age_minutes: float = 30.0) -> bool:
        """Check if baseline is too old to be reliable."""
        return self.get_baseline_age_minutes() > max_age_minutes

    def get_resource_summary(self) -> dict[str, Any]:
        """Get summary of baseline resource state."""
        return {
            "memory_mb": self.memory_mb,
            "processes": self.process_count,
            "file_handles": self.file_handles,
            "gpu_memory_mb": self.gpu_memory_mb,
            "cpu_percent": self.cpu_percent,
            "threads": self.thread_count,
            "age_minutes": self.get_baseline_age_minutes(),
        }


@dataclass
class CleanupValidationConfig:
    """Configuration for resource cleanup validation."""

    # Tolerance thresholds
    memory_leak_tolerance_percent: float = 5.0
    gpu_memory_leak_tolerance_percent: float = 2.0
    cpu_usage_tolerance_percent: float = 10.0

    # Validation timeouts
    cleanup_timeout_seconds: float = 30.0
    baseline_establishment_timeout_seconds: float = 10.0

    # Resource monitoring settings
    enable_gpu_monitoring: bool = True
    enable_file_handle_monitoring: bool = True
    enable_process_monitoring: bool = True

    # Cleanup strategies
    force_garbage_collection: bool = True
    force_cuda_cache_clear: bool = True
    verify_temp_file_cleanup: bool = True

    # Validation scope
    validate_workflow_components: bool = True
    validate_memory_cleanup: bool = True
    validate_process_cleanup: bool = True
    validate_baseline_restoration: bool = True

    def validate_config(self) -> list[str]:
        """Validate configuration values."""
        errors = []

        if self.memory_leak_tolerance_percent < 0:
            errors.append("Memory leak tolerance must be non-negative")

        if self.cleanup_timeout_seconds <= 0:
            errors.append("Cleanup timeout must be positive")

        if self.baseline_establishment_timeout_seconds <= 0:
            errors.append("Baseline establishment timeout must be positive")

        return errors


class ResourceCleanupValidator(Protocol):
    """Protocol for resource cleanup validation components."""

    def establish_resource_baseline(
        self, workflow_id: str
    ) -> ResourceBaseline:
        """Establish baseline resource state before workflow execution."""
        ...

    def validate_resource_cleanup(
        self, workflow_id: str, cleanup_operation: Callable[[], Any]
    ) -> ResourceCleanupMetrics:
        """Validate resource cleanup after workflow execution."""
        ...

    def get_cleanup_metrics(self) -> list[ResourceCleanupMetrics]:
        """Get all collected cleanup metrics."""
        ...

    def clear_cleanup_history(self) -> None:
        """Clear cleanup metrics history."""
        ...


class ResourceMonitor(Protocol):
    """Protocol for resource monitoring components."""

    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in megabytes."""
        ...

    def get_process_count(self) -> int:
        """Get current process count."""
        ...

    def get_file_handle_count(self) -> int:
        """Get current file handle count."""
        ...

    def get_gpu_memory_usage_mb(self) -> float:
        """Get current GPU memory usage in megabytes."""
        ...

    def get_cpu_usage_percent(self) -> float:
        """Get current CPU usage percentage."""
        ...

    def get_open_ports(self) -> list[int]:
        """Get list of open network ports."""
        ...

    def scan_temp_files(self) -> list[Path]:
        """Scan for temporary files that may need cleanup."""
        ...


@dataclass
class CleanupValidationReport:
    """Report for resource cleanup validation results."""

    validation_timestamp: datetime
    total_workflows_tested: int
    successful_cleanups: int
    failed_cleanups: int
    memory_leaks_detected: int
    process_leaks_detected: int
    file_leaks_detected: int
    gpu_leaks_detected: int

    avg_cleanup_time_seconds: float
    max_cleanup_time_seconds: float
    min_cleanup_time_seconds: float

    cleanup_metrics: list[ResourceCleanupMetrics] = field(default_factory=list)

    @property
    def success_rate_percent(self) -> float:
        """Calculate overall cleanup success rate."""
        if self.total_workflows_tested == 0:
            return 0.0
        return (self.successful_cleanups / self.total_workflows_tested) * 100.0

    @property
    def memory_leak_rate_percent(self) -> float:
        """Calculate memory leak detection rate."""
        if self.total_workflows_tested == 0:
            return 0.0
        return (
            self.memory_leaks_detected / self.total_workflows_tested
        ) * 100.0

    def get_summary_dict(self) -> dict[str, Any]:
        """Get summary as dictionary for serialization."""
        return {
            "validation_timestamp": self.validation_timestamp.isoformat(),
            "total_workflows_tested": self.total_workflows_tested,
            "success_rate_percent": self.success_rate_percent,
            "memory_leak_rate_percent": self.memory_leak_rate_percent,
            "avg_cleanup_time_seconds": self.avg_cleanup_time_seconds,
            "leak_breakdown": {
                "memory_leaks": self.memory_leaks_detected,
                "process_leaks": self.process_leaks_detected,
                "file_leaks": self.file_leaks_detected,
                "gpu_leaks": self.gpu_leaks_detected,
            },
        }
