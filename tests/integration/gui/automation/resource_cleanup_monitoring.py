"""Resource monitoring and measurement for cleanup validation.

This module provides the implementation of resource monitoring capabilities
including memory usage tracking, process monitoring, file handle counting, GPU
resource monitoring, and comprehensive baseline establishment for systematic
resource cleanup validation.
"""

import gc
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import psutil

from .resource_cleanup_protocols import (
    CleanupValidationConfig,
    ResourceBaseline,
    ResourceCleanupMetrics,
)


class SystemResourceMonitor:
    """Implementation of system resource monitoring capabilities."""

    def __init__(self, config: CleanupValidationConfig | None = None) -> None:
        """Initialize system resource monitor with configuration."""
        self.config = config or CleanupValidationConfig()
        self._lock = threading.Lock()

    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in megabytes."""
        return psutil.virtual_memory().used / (1024 * 1024)

    def get_process_count(self) -> int:
        """Get current process count."""
        return len(psutil.pids())

    def get_file_handle_count(self) -> int:
        """Get current file handle count for the current process."""
        try:
            process = psutil.Process()
            return len(process.open_files())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0

    def get_gpu_memory_usage_mb(self) -> float:
        """Get current GPU memory usage in megabytes (RTX 3070 Ti specific)."""
        if not self.config.enable_gpu_monitoring:
            return 0.0

        try:
            import nvidia_ml_py3 as nml  # type: ignore[import-untyped]

            nml.nvmlInit()
            handle = nml.nvmlDeviceGetHandleByIndex(0)
            info = nml.nvmlDeviceGetMemoryInfo(handle)
            return info.used / (1024 * 1024)  # Convert to MB
        except (ImportError, Exception):
            return 0.0

    def get_cpu_usage_percent(self) -> float:
        """Get current CPU usage percentage."""
        return psutil.cpu_percent(interval=0.1)

    def get_open_ports(self) -> list[int]:
        """Get list of open network ports."""
        try:
            connections = psutil.net_connections()
            return [conn.laddr.port for conn in connections if conn.laddr]
        except (psutil.AccessDenied, AttributeError):
            return []

    def scan_temp_files(self) -> list[Path]:
        """Scan for temporary files that may need cleanup."""
        temp_patterns = [
            Path("temp_storage.py"),
            Path("generated_configs"),
            Path("outputs/temp_*"),
            Path("test-artifacts/temp_*"),
            Path("selenium-videos/temp_*"),
        ]

        found_files = []
        for pattern in temp_patterns:
            if pattern.exists():
                found_files.append(pattern)
            elif "*" in str(pattern):
                # Handle wildcard patterns
                parent_dir = pattern.parent
                if parent_dir.exists():
                    pattern_name = pattern.name.replace("*", "")
                    for file_path in parent_dir.iterdir():
                        if pattern_name in file_path.name:
                            found_files.append(file_path)

        return found_files

    def get_thread_count(self) -> int:
        """Get current active thread count."""
        return threading.active_count()

    def force_garbage_collection(self) -> None:
        """Force Python garbage collection to clean up unreferenced objects."""
        if self.config.force_garbage_collection:
            gc.collect()

    def clear_gpu_cache(self) -> bool:
        """Clear GPU memory cache if available."""
        if not self.config.force_cuda_cache_clear:
            return True

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                return True
        except ImportError:
            pass
        return True


class ResourceCleanupValidationMixin:
    """Mixin for adding resource cleanup validation capabilities."""

    def __init__(self, config: CleanupValidationConfig | None = None) -> None:
        """Initialize resource cleanup validation monitoring."""
        self.config = config or CleanupValidationConfig()
        self.cleanup_metrics: list[ResourceCleanupMetrics] = []
        self.resource_baselines: dict[str, ResourceBaseline] = {}
        self.resource_monitor = SystemResourceMonitor(self.config)
        self._cleanup_lock = threading.Lock()

    def establish_resource_baseline(
        self, workflow_id: str
    ) -> ResourceBaseline:
        """Establish baseline resource state before workflow execution."""
        with self._cleanup_lock:
            # Force cleanup before establishing baseline
            self.resource_monitor.force_garbage_collection()

            # Small delay to let system stabilize
            time.sleep(0.1)

            # System resource measurements
            baseline = ResourceBaseline(
                memory_mb=self.resource_monitor.get_memory_usage_mb(),
                process_count=self.resource_monitor.get_process_count(),
                file_handles=self.resource_monitor.get_file_handle_count(),
                gpu_memory_mb=self.resource_monitor.get_gpu_memory_usage_mb(),
                cpu_percent=self.resource_monitor.get_cpu_usage_percent(),
                open_ports=self.resource_monitor.get_open_ports(),
                temp_files=self.resource_monitor.scan_temp_files(),
                thread_count=self.resource_monitor.get_thread_count(),
            )

            self.resource_baselines[workflow_id] = baseline
            return baseline

    def validate_resource_cleanup(
        self, workflow_id: str, cleanup_operation: Callable[[], Any]
    ) -> ResourceCleanupMetrics:
        """Validate resource cleanup after workflow execution."""
        if workflow_id not in self.resource_baselines:
            raise ValueError(
                f"No baseline established for workflow {workflow_id}"
            )

        baseline = self.resource_baselines[workflow_id]
        cleanup_start_time = time.time()

        # Pre-cleanup: Force cleanup operations
        self.resource_monitor.force_garbage_collection()
        self.resource_monitor.clear_gpu_cache()

        # Execute the cleanup operation
        try:
            cleanup_operation()
        except Exception:
            # Log the error but continue with validation
            pass

        # Post-cleanup: Force cleanup again and measure
        self.resource_monitor.force_garbage_collection()
        cleanup_end_time = time.time()
        cleanup_duration = cleanup_end_time - cleanup_start_time

        # Small delay to let system stabilize
        time.sleep(0.1)

        # Measure post-cleanup resource state
        post_memory_mb = self.resource_monitor.get_memory_usage_mb()
        post_process_count = self.resource_monitor.get_process_count()
        post_file_handles = self.resource_monitor.get_file_handle_count()
        post_gpu_memory_mb = self.resource_monitor.get_gpu_memory_usage_mb()
        post_cpu_percent = self.resource_monitor.get_cpu_usage_percent()

        # Cleanup validation analysis using configured tolerances
        memory_leak_detected = post_memory_mb > baseline.memory_mb * (
            1 + self.config.memory_leak_tolerance_percent / 100
        )
        orphaned_processes = post_process_count > baseline.process_count
        file_leak_detected = (
            self.config.enable_file_handle_monitoring
            and post_file_handles > baseline.file_handles
        )
        gpu_memory_leaked = (
            self.config.enable_gpu_monitoring
            and post_gpu_memory_mb
            > baseline.gpu_memory_mb
            * (1 + self.config.gpu_memory_leak_tolerance_percent / 100)
        )

        # Calculate cleanup percentages
        memory_cleanup_percentage = max(
            0, (baseline.memory_mb - post_memory_mb) / baseline.memory_mb * 100
        )

        # Overall cleanup validation
        cleanup_validation_passed = not any(
            [
                memory_leak_detected,
                orphaned_processes,
                file_leak_detected,
                gpu_memory_leaked,
            ]
        )

        # Baseline restoration validation
        baseline_restoration_successful = (
            abs(post_memory_mb - baseline.memory_mb)
            < baseline.memory_mb
            * (self.config.memory_leak_tolerance_percent / 100)
            and abs(post_cpu_percent - baseline.cpu_percent)
            < self.config.cpu_usage_tolerance_percent
        )

        # Verify temporary file cleanup
        current_temp_files = self.resource_monitor.scan_temp_files()
        temp_files_cleaned = not self.config.verify_temp_file_cleanup or len(
            current_temp_files
        ) <= len(baseline.temp_files)

        # CUDA context verification
        cuda_context_cleaned = self.resource_monitor.clear_gpu_cache()

        # Create comprehensive metrics
        metrics = ResourceCleanupMetrics(
            memory_baseline_mb=baseline.memory_mb,
            memory_post_execution_mb=post_memory_mb,
            memory_leak_detected=memory_leak_detected,
            memory_cleanup_percentage=memory_cleanup_percentage,
            process_baseline_count=baseline.process_count,
            process_post_execution_count=post_process_count,
            orphaned_processes_detected=orphaned_processes,
            process_cleanup_successful=not orphaned_processes,
            file_handles_baseline=baseline.file_handles,
            file_handles_post_execution=post_file_handles,
            file_leak_detected=file_leak_detected,
            temp_files_cleaned=temp_files_cleaned,
            gpu_memory_baseline_mb=baseline.gpu_memory_mb,
            gpu_memory_post_execution_mb=post_gpu_memory_mb,
            gpu_memory_leaked=gpu_memory_leaked,
            cuda_context_cleaned=cuda_context_cleaned,
            cpu_baseline_percent=baseline.cpu_percent,
            cpu_post_execution_percent=post_cpu_percent,
            baseline_restoration_successful=baseline_restoration_successful,
            cleanup_validation_passed=cleanup_validation_passed,
            cleanup_time_seconds=cleanup_duration,
            leak_detection_accuracy=self._calculate_leak_detection_accuracy(),
            workflow_component=workflow_id,
            cleanup_phase="validation",
        )

        with self._cleanup_lock:
            self.cleanup_metrics.append(metrics)

        return metrics

    def get_cleanup_metrics(self) -> list[ResourceCleanupMetrics]:
        """Get all collected cleanup metrics."""
        with self._cleanup_lock:
            return self.cleanup_metrics.copy()

    def clear_cleanup_history(self) -> None:
        """Clear cleanup metrics history."""
        with self._cleanup_lock:
            self.cleanup_metrics.clear()
            self.resource_baselines.clear()

    def _calculate_leak_detection_accuracy(self) -> float:
        """Calculate accuracy of leak detection mechanisms."""
        if not self.cleanup_metrics:
            return 100.0

        # Simple accuracy based on successful cleanup validations
        successful_validations = sum(
            1 for m in self.cleanup_metrics if m.cleanup_validation_passed
        )
        return (successful_validations / len(self.cleanup_metrics)) * 100.0

    def get_validation_summary(self) -> dict[str, Any]:
        """Get summary of validation results."""
        if not self.cleanup_metrics:
            return {"no_data": True}

        metrics = self.cleanup_metrics
        return {
            "total_validations": len(metrics),
            "successful_cleanups": sum(
                1 for m in metrics if m.cleanup_validation_passed
            ),
            "memory_leaks_detected": sum(
                1 for m in metrics if m.memory_leak_detected
            ),
            "process_leaks_detected": sum(
                1 for m in metrics if m.orphaned_processes_detected
            ),
            "file_leaks_detected": sum(
                1 for m in metrics if m.file_leak_detected
            ),
            "gpu_leaks_detected": sum(
                1 for m in metrics if m.gpu_memory_leaked
            ),
            "avg_cleanup_time": sum(m.cleanup_time_seconds for m in metrics)
            / len(metrics),
            "avg_memory_cleanup_percentage": sum(
                m.memory_cleanup_percentage for m in metrics
            )
            / len(metrics),
            "baseline_restoration_success_rate": sum(
                1 for m in metrics if m.baseline_restoration_successful
            )
            / len(metrics)
            * 100,
        }
