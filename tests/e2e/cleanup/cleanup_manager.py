"""Cleanup Manager with ResourceMonitor Integration.

This module provides the main coordination for resource cleanup automation,
integrating with the existing ResourceMonitor system for real-time validation
and comprehensive cleanup orchestration between test runs.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.utils.monitoring import ResourceMonitor, ResourceSnapshot

logger = logging.getLogger(__name__)


class CleanupStatus(Enum):
    """Status enumeration for cleanup operations."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    ROLLBACK_REQUIRED = "rollback_required"


@dataclass
class CleanupConfig:
    """Configuration for cleanup operations."""

    # Resource monitoring
    enable_resource_monitoring: bool = True
    monitoring_interval: float = 0.5
    validation_timeout: float = 30.0

    # Cleanup procedures
    cleanup_timeout: float = 60.0
    force_cleanup: bool = True
    enable_rollback: bool = True

    # Resource thresholds for validation
    max_memory_leak_mb: float = 100.0
    max_process_leak_count: int = 5
    max_file_handle_leak_count: int = 20
    max_temp_files_count: int = 50

    # Retry configuration
    max_retries: int = 3
    retry_delay: float = 2.0

    # Paths
    temp_cleanup_patterns: list[str] = field(
        default_factory=lambda: [
            "temp_*",
            "*.tmp",
            "test_output_*",
            "crackseg_test_*",
        ]
    )


@dataclass
class CleanupResult:
    """Result of cleanup operation execution."""

    status: CleanupStatus
    duration_seconds: float
    resources_cleaned: dict[str, int]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # Resource monitoring data
    baseline_snapshot: ResourceSnapshot | None = None
    post_cleanup_snapshot: ResourceSnapshot | None = None
    resource_leak_detected: bool = False

    # Validation details
    validation_passed: bool = True
    rollback_performed: bool = False


class CleanupManager:
    """Main coordinator for resource cleanup with ResourceMonitor integration.

    Orchestrates cleanup procedures with real-time resource monitoring
    and validation to ensure complete environment reset between test runs.
    """

    def __init__(self, config: CleanupConfig | None = None) -> None:
        """Initialize cleanup manager with configuration."""
        self.config = config or CleanupConfig()

        # Resource monitoring integration
        self.resource_monitor: ResourceMonitor | None = None
        self.baseline_snapshot: ResourceSnapshot | None = None

        # State tracking
        self._cleanup_active = False
        self._cleanup_registry: dict[str, Any] = {}

        self.logger = logging.getLogger(__name__)

    async def establish_baseline(self) -> ResourceSnapshot:
        """Establish resource baseline before test execution."""
        self.logger.info(
            "Establishing resource baseline for cleanup validation"
        )

        # Initialize resource monitoring if enabled
        if self.config.enable_resource_monitoring:
            self.resource_monitor = ResourceMonitor(
                enable_gpu_monitoring=True,
                enable_network_monitoring=True,
                enable_file_monitoring=True,
            )

            # Start monitoring for baseline establishment
            self.resource_monitor.start_real_time_monitoring(
                interval=self.config.monitoring_interval
            )

            # Wait for stabilization
            await asyncio.sleep(1.0)

            # Capture baseline
            self.baseline_snapshot = (
                self.resource_monitor.get_current_snapshot()
            )

            self.logger.info(
                f"Baseline established: {self.baseline_snapshot.get_summary()}"
            )

            return self.baseline_snapshot

        # Fallback without monitoring
        self.logger.warning(
            "Resource monitoring disabled, no baseline captured"
        )
        return ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=0.0,
            memory_used_mb=0.0,
            memory_available_mb=0.0,
            memory_percent=0.0,
            gpu_memory_used_mb=0.0,
            gpu_memory_total_mb=8192.0,  # RTX 3070 Ti default
            gpu_memory_percent=0.0,
            gpu_utilization_percent=0.0,
            gpu_temperature_celsius=0.0,
            process_count=0,
            thread_count=0,
            file_handles=0,
            network_connections=0,
            open_ports=[],
            disk_read_mb=0.0,
            disk_write_mb=0.0,
            temp_files_count=0,
            temp_files_size_mb=0.0,
        )

    async def execute_cleanup(
        self, test_id: str, cleanup_procedures: list[str] | None = None
    ) -> CleanupResult:
        """Execute comprehensive cleanup with validation."""
        if self._cleanup_active:
            raise RuntimeError("Cleanup operation already in progress")

        self._cleanup_active = True
        start_time = time.time()

        try:
            self.logger.info(f"Starting cleanup for test {test_id}")

            # Default cleanup procedures if not specified
            if cleanup_procedures is None:
                cleanup_procedures = [
                    "temp_files",
                    "processes",
                    "network_connections",
                    "file_handles",
                    "gpu_cache",
                ]

            # Execute cleanup with timeout
            result = await asyncio.wait_for(
                self._perform_cleanup(test_id, cleanup_procedures),
                timeout=self.config.cleanup_timeout,
            )

            # Calculate duration
            result.duration_seconds = time.time() - start_time

            self.logger.info(
                f"Cleanup completed for {test_id}: {result.status.value}"
            )

            return result

        except TimeoutError:
            self.logger.error(f"Cleanup timeout exceeded for {test_id}")
            return CleanupResult(
                status=CleanupStatus.FAILED,
                duration_seconds=time.time() - start_time,
                resources_cleaned={},
                errors=["Cleanup operation timed out"],
            )
        except Exception as e:
            self.logger.error(f"Cleanup failed for {test_id}: {e}")
            return CleanupResult(
                status=CleanupStatus.FAILED,
                duration_seconds=time.time() - start_time,
                resources_cleaned={},
                errors=[str(e)],
            )
        finally:
            self._cleanup_active = False

    async def _perform_cleanup(
        self, test_id: str, procedures: list[str]
    ) -> CleanupResult:
        """Perform actual cleanup operations with monitoring."""
        resources_cleaned: dict[str, int] = {}
        errors: list[str] = []
        warnings: list[str] = []

        # Import cleanup procedures dynamically to avoid circular imports
        from .resource_cleanup import ResourceCleanupRegistry

        registry = ResourceCleanupRegistry()

        # Execute each cleanup procedure
        for procedure_name in procedures:
            try:
                self.logger.debug(
                    f"Executing cleanup procedure: {procedure_name}"
                )

                # Get cleanup procedure
                if procedure_name not in registry.get_available_procedures():
                    warnings.append(
                        f"Unknown cleanup procedure: {procedure_name}"
                    )
                    continue

                # Execute cleanup
                cleaned_count = await registry.execute_cleanup(
                    procedure_name, test_id=test_id
                )
                resources_cleaned[procedure_name] = cleaned_count

            except Exception as e:
                error_msg = f"Failed to execute {procedure_name}: {e}"
                self.logger.error(error_msg)
                errors.append(error_msg)

        # Capture post-cleanup snapshot
        post_cleanup_snapshot = None
        if self.resource_monitor:
            await asyncio.sleep(0.5)  # Allow system to stabilize
            post_cleanup_snapshot = (
                self.resource_monitor.get_current_snapshot()
            )

        # Validate cleanup effectiveness
        leak_detected = self._detect_resource_leaks(post_cleanup_snapshot)

        # Determine overall status
        status = self._determine_cleanup_status(
            resources_cleaned, errors, leak_detected
        )

        return CleanupResult(
            status=status,
            duration_seconds=0.0,  # Will be set by caller
            resources_cleaned=resources_cleaned,
            errors=errors,
            warnings=warnings,
            baseline_snapshot=self.baseline_snapshot,
            post_cleanup_snapshot=post_cleanup_snapshot,
            resource_leak_detected=leak_detected,
            validation_passed=not leak_detected and len(errors) == 0,
        )

    def _detect_resource_leaks(
        self, post_snapshot: ResourceSnapshot | None
    ) -> bool:
        """Detect resource leaks by comparing snapshots."""
        if not self.baseline_snapshot or not post_snapshot:
            return False

        # Memory leak detection
        memory_diff = (
            post_snapshot.memory_used_mb
            - self.baseline_snapshot.memory_used_mb
        )
        if memory_diff > self.config.max_memory_leak_mb:
            self.logger.warning(f"Memory leak detected: {memory_diff:.2f}MB")
            return True

        # Process leak detection
        process_diff = (
            post_snapshot.process_count - self.baseline_snapshot.process_count
        )
        if process_diff > self.config.max_process_leak_count:
            self.logger.warning(
                f"Process leak detected: {process_diff} processes"
            )
            return True

        # File handle leak detection
        if hasattr(post_snapshot, "file_handles") and hasattr(
            self.baseline_snapshot, "file_handles"
        ):
            handle_diff = (
                post_snapshot.file_handles
                - self.baseline_snapshot.file_handles
            )
            if handle_diff > self.config.max_file_handle_leak_count:
                self.logger.warning(
                    f"File handle leak detected: {handle_diff} handles"
                )
                return True

        return False

    def _determine_cleanup_status(
        self,
        resources_cleaned: dict[str, int],
        errors: list[str],
        leak_detected: bool,
    ) -> CleanupStatus:
        """Determine overall cleanup status from results."""
        if errors:
            return (
                CleanupStatus.FAILED
                if leak_detected
                else CleanupStatus.PARTIAL
            )

        if leak_detected:
            return CleanupStatus.ROLLBACK_REQUIRED

        if any(count > 0 for count in resources_cleaned.values()):
            return CleanupStatus.SUCCESS

        return CleanupStatus.SUCCESS  # No resources to clean is also success

    async def shutdown(self) -> None:
        """Shutdown cleanup manager and stop monitoring."""
        if self.resource_monitor:
            self.resource_monitor.stop_monitoring()
            self.resource_monitor = None

        self._cleanup_active = False
        self.logger.info("Cleanup manager shutdown completed")
