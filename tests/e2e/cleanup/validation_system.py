"""Cleanup Validation System with Rollback Capabilities.

This module provides comprehensive validation of cleanup operations
with rollback capabilities when cleanup fails or resource leaks are detected.
Integrates with ResourceMonitor for real-time validation.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

from src.utils.monitoring import ResourceSnapshot

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Status enumeration for validation operations."""

    PENDING = "pending"
    VALIDATING = "validating"
    PASSED = "passed"
    FAILED = "failed"
    ROLLBACK_REQUIRED = "rollback_required"
    ROLLBACK_COMPLETED = "rollback_completed"


@dataclass
class ValidationResult:
    """Result of cleanup validation operation."""

    status: ValidationStatus
    validation_duration: float
    leak_detected: bool
    resource_differences: dict[str, float]
    validation_errors: list[str] = field(default_factory=list)
    rollback_performed: bool = False
    rollback_success: bool = False


class CleanupValidator:
    """Validates cleanup effectiveness and detects resource leaks."""

    def __init__(
        self, tolerance_config: dict[str, float] | None = None
    ) -> None:
        """Initialize cleanup validator with tolerance configuration."""
        self.tolerance_config = tolerance_config or {
            "memory_mb": 50.0,
            "process_count": 2,
            "file_handles": 10,
            "gpu_memory_mb": 100.0,
            "temp_files": 5,
        }

        self.logger = logging.getLogger(__name__)

    async def validate_cleanup(
        self,
        baseline_snapshot: ResourceSnapshot,
        post_cleanup_snapshot: ResourceSnapshot,
        test_id: str,
    ) -> ValidationResult:
        """Validate cleanup effectiveness by comparing resource snapshots."""
        start_time = time.time()

        self.logger.info(f"Starting cleanup validation for test {test_id}")

        try:
            # Calculate resource differences
            differences = self._calculate_resource_differences(
                baseline_snapshot, post_cleanup_snapshot
            )

            # Check for resource leaks
            leak_detected = self._detect_leaks(differences)

            # Determine validation status
            status = (
                ValidationStatus.PASSED
                if not leak_detected
                else ValidationStatus.FAILED
            )

            validation_duration = time.time() - start_time

            result = ValidationResult(
                status=status,
                validation_duration=validation_duration,
                leak_detected=leak_detected,
                resource_differences=differences,
                validation_errors=[],
            )

            if leak_detected:
                self.logger.warning(
                    f"Resource leaks detected for test {test_id}: "
                    f"{differences}"
                )
                result.status = ValidationStatus.ROLLBACK_REQUIRED
            else:
                self.logger.info(
                    f"Cleanup validation passed for test {test_id}"
                )

            return result

        except Exception as e:
            error_msg = f"Validation failed for test {test_id}: {e}"
            self.logger.error(error_msg)

            return ValidationResult(
                status=ValidationStatus.FAILED,
                validation_duration=time.time() - start_time,
                leak_detected=True,
                resource_differences={},
                validation_errors=[error_msg],
            )

    def _calculate_resource_differences(
        self, baseline: ResourceSnapshot, post_cleanup: ResourceSnapshot
    ) -> dict[str, float]:
        """Calculate differences between baseline and post-cleanup states."""
        differences = {
            "memory_mb": post_cleanup.memory_used_mb - baseline.memory_used_mb,
            "process_count": float(
                post_cleanup.process_count - baseline.process_count
            ),
            "cpu_percent": post_cleanup.cpu_percent - baseline.cpu_percent,
        }

        # Add GPU memory if available
        if hasattr(post_cleanup, "gpu_memory_used_mb") and hasattr(
            baseline, "gpu_memory_used_mb"
        ):
            differences["gpu_memory_mb"] = (
                post_cleanup.gpu_memory_used_mb - baseline.gpu_memory_used_mb
            )

        # Add file handles if available
        if hasattr(post_cleanup, "file_handles") and hasattr(
            baseline, "file_handles"
        ):
            differences["file_handles"] = float(
                post_cleanup.file_handles - baseline.file_handles
            )

        # Add temp files if available
        if hasattr(post_cleanup, "temp_files_count") and hasattr(
            baseline, "temp_files_count"
        ):
            differences["temp_files"] = float(
                post_cleanup.temp_files_count - baseline.temp_files_count
            )

        return differences

    def _detect_leaks(self, differences: dict[str, float]) -> bool:
        """Detect resource leaks based on tolerance thresholds."""
        for resource_type, difference in differences.items():
            tolerance = self.tolerance_config.get(resource_type, 0.0)

            if difference > tolerance:
                self.logger.warning(
                    f"Resource leak detected - {resource_type}: "
                    f"{difference:.2f} (tolerance: {tolerance})"
                )
                return True

        return False


class RollbackManager:
    """Manages rollback operations when cleanup fails."""

    def __init__(self) -> None:
        """Initialize rollback manager."""
        self.logger = logging.getLogger(__name__)
        self._rollback_snapshots: dict[str, ResourceSnapshot] = {}

    async def perform_rollback(
        self,
        test_id: str,
        baseline_snapshot: ResourceSnapshot,
        failed_cleanup_procedures: list[str],
    ) -> bool:
        """Perform rollback operations to restore baseline state."""
        self.logger.info(f"Starting rollback for test {test_id}")

        try:
            # Store rollback point
            self._rollback_snapshots[test_id] = baseline_snapshot

            # Execute rollback procedures
            rollback_success = await self._execute_rollback_procedures(
                test_id, failed_cleanup_procedures
            )

            if rollback_success:
                self.logger.info(
                    f"Rollback completed successfully for test {test_id}"
                )
            else:
                self.logger.error(f"Rollback failed for test {test_id}")

            return rollback_success

        except Exception as e:
            self.logger.error(
                f"Rollback operation failed for test {test_id}: {e}"
            )
            return False

    async def _execute_rollback_procedures(
        self, test_id: str, failed_procedures: list[str]
    ) -> bool:
        """Execute specific rollback procedures."""
        rollback_success = True

        # Import cleanup registry dynamically
        from .resource_cleanup import ResourceCleanupRegistry

        registry = ResourceCleanupRegistry()

        # Execute rollback for each failed procedure
        for procedure_name in failed_procedures:
            try:
                self.logger.info(f"Rolling back procedure: {procedure_name}")

                # For rollback, we want to re-run cleanup more aggressively
                await registry.execute_cleanup(
                    procedure_name, f"rollback_{test_id}"
                )

            except Exception as e:
                self.logger.error(
                    f"Rollback failed for procedure {procedure_name}: {e}"
                )
                rollback_success = False

        # Additional system-level rollback operations
        try:
            # Force garbage collection
            import gc

            gc.collect()

            # Clear any remaining caches
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except ImportError:
            # PyTorch not available, skip GPU cleanup
            pass
        except Exception as e:
            self.logger.warning(f"System-level rollback warning: {e}")

        return rollback_success

    def clear_rollback_history(self, test_id: str | None = None) -> None:
        """Clear rollback history for specific test or all tests."""
        if test_id:
            self._rollback_snapshots.pop(test_id, None)
            self.logger.debug(f"Cleared rollback history for test {test_id}")
        else:
            self._rollback_snapshots.clear()
            self.logger.debug("Cleared all rollback history")

    def get_rollback_snapshot(self, test_id: str) -> ResourceSnapshot | None:
        """Get stored rollback snapshot for a test."""
        return self._rollback_snapshots.get(test_id)


async def validate_and_rollback(
    baseline_snapshot: ResourceSnapshot,
    post_cleanup_snapshot: ResourceSnapshot,
    test_id: str,
    failed_procedures: list[str] | None = None,
) -> ValidationResult:
    """Convenience function for validation with automatic rollback."""
    validator = CleanupValidator()
    rollback_manager = RollbackManager()

    # Perform validation
    result = await validator.validate_cleanup(
        baseline_snapshot, post_cleanup_snapshot, test_id
    )

    # Perform rollback if required
    if result.status == ValidationStatus.ROLLBACK_REQUIRED:
        rollback_success = await rollback_manager.perform_rollback(
            test_id, baseline_snapshot, failed_procedures or []
        )

        result.rollback_performed = True
        result.rollback_success = rollback_success

        if rollback_success:
            result.status = ValidationStatus.ROLLBACK_COMPLETED

    return result


# Subtask 16.7 - Enhanced Cleanup Validation System
# Note: Implementation moved to separate modules to respect file size limits
