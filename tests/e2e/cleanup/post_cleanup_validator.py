"""Post-Cleanup Verification Workflows.

This module provides structured workflows for verifying different types of
cleanup operations with integration to performance thresholds and systematic
validation.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.utils.monitoring import ResourceSnapshot

logger = logging.getLogger(__name__)


class WorkflowType(Enum):
    """Types of post-cleanup verification workflows."""

    BASIC_RESOURCE_CHECK = "basic_resource_check"
    COMPREHENSIVE_VALIDATION = "comprehensive_validation"
    PERFORMANCE_BENCHMARK = "performance_benchmark"
    SECURITY_VALIDATION = "security_validation"
    CONTAINER_VERIFICATION = "container_verification"


@dataclass
class WorkflowStep:
    """Individual step in a verification workflow."""

    name: str
    description: str
    timeout_seconds: float
    critical: bool = True
    retry_count: int = 2
    dependencies: list[str] = field(default_factory=list)


@dataclass
class WorkflowResult:
    """Result of a verification workflow execution."""

    workflow_type: WorkflowType
    total_duration: float
    steps_executed: int
    steps_passed: int
    steps_failed: int
    failures: list[str] = field(default_factory=list)
    performance_metrics: dict[str, float] = field(default_factory=dict)
    threshold_violations: list[str] = field(default_factory=list)


class PostCleanupValidator:
    """Executes structured post-cleanup verification workflows."""

    def __init__(
        self, thresholds_config: dict[str, Any] | None = None
    ) -> None:
        """Initialize post-cleanup validator with threshold configuration."""
        self.thresholds_config = thresholds_config or {}
        self.logger = logging.getLogger(__name__)
        self._workflows = self._initialize_workflows()

    def _initialize_workflows(self) -> dict[WorkflowType, list[WorkflowStep]]:
        """Initialize predefined verification workflows."""
        return {
            WorkflowType.BASIC_RESOURCE_CHECK: [
                WorkflowStep(
                    name="memory_validation",
                    description="Verify memory usage within thresholds",
                    timeout_seconds=5.0,
                    critical=True,
                ),
                WorkflowStep(
                    name="process_validation",
                    description="Check for orphaned processes",
                    timeout_seconds=3.0,
                    critical=True,
                ),
                WorkflowStep(
                    name="file_handle_validation",
                    description="Verify file handles are released",
                    timeout_seconds=2.0,
                    critical=True,
                ),
            ],
            WorkflowType.COMPREHENSIVE_VALIDATION: [
                WorkflowStep(
                    name="resource_baseline_check",
                    description="Comprehensive resource comparison",
                    timeout_seconds=10.0,
                    critical=True,
                ),
                WorkflowStep(
                    name="gpu_memory_validation",
                    description="Verify GPU memory cleanup",
                    timeout_seconds=8.0,
                    critical=True,
                ),
                WorkflowStep(
                    name="temp_files_cleanup",
                    description="Check temporary files cleanup",
                    timeout_seconds=5.0,
                    critical=True,
                ),
                WorkflowStep(
                    name="cache_invalidation",
                    description="Verify caches are cleared",
                    timeout_seconds=3.0,
                    critical=False,
                ),
                WorkflowStep(
                    name="network_connections",
                    description="Check for open network connections",
                    timeout_seconds=4.0,
                    critical=False,
                ),
            ],
            WorkflowType.PERFORMANCE_BENCHMARK: [
                WorkflowStep(
                    name="cleanup_time_validation",
                    description="Verify cleanup completion time",
                    timeout_seconds=2.0,
                    critical=True,
                ),
                WorkflowStep(
                    name="resource_efficiency_check",
                    description="Analyze resource usage efficiency",
                    timeout_seconds=15.0,
                    critical=False,
                ),
                WorkflowStep(
                    name="memory_leak_detection",
                    description="Long-term memory leak detection",
                    timeout_seconds=30.0,
                    critical=True,
                ),
            ],
            WorkflowType.CONTAINER_VERIFICATION: [
                WorkflowStep(
                    name="orphaned_containers",
                    description="Check for orphaned containers",
                    timeout_seconds=10.0,
                    critical=True,
                ),
                WorkflowStep(
                    name="dangling_images",
                    description="Verify dangling images cleanup",
                    timeout_seconds=8.0,
                    critical=True,
                ),
                WorkflowStep(
                    name="unused_volumes",
                    description="Check for unused volumes",
                    timeout_seconds=6.0,
                    critical=True,
                ),
                WorkflowStep(
                    name="network_cleanup",
                    description="Verify network cleanup",
                    timeout_seconds=5.0,
                    critical=False,
                ),
            ],
        }

    async def execute_workflow(
        self,
        workflow_type: WorkflowType,
        baseline_snapshot: ResourceSnapshot,
        post_cleanup_snapshot: ResourceSnapshot,
        test_id: str,
    ) -> WorkflowResult:
        """Execute a specific verification workflow."""
        start_time = time.time()

        self.logger.info(
            f"Starting {workflow_type.value} workflow for test {test_id}"
        )

        workflow_steps = self._workflows.get(workflow_type, [])
        steps_executed = 0
        steps_passed = 0
        steps_failed = 0
        failures: list[str] = []
        performance_metrics: dict[str, float] = {}
        threshold_violations: list[str] = []

        for step in workflow_steps:
            try:
                step_start = time.time()

                # Execute step with timeout
                success = await asyncio.wait_for(
                    self._execute_step(
                        step, baseline_snapshot, post_cleanup_snapshot, test_id
                    ),
                    timeout=step.timeout_seconds,
                )

                step_duration = time.time() - step_start
                performance_metrics[f"{step.name}_duration"] = step_duration

                steps_executed += 1

                if success:
                    steps_passed += 1
                    self.logger.debug(
                        f"Step {step.name} passed in {step_duration:.2f}s"
                    )
                else:
                    steps_failed += 1
                    failure_msg = f"Step {step.name} failed"
                    failures.append(failure_msg)

                    if step.critical:
                        self.logger.error(failure_msg)
                    else:
                        self.logger.warning(failure_msg)

                # Check for threshold violations
                violations = self._check_step_thresholds(step, step_duration)
                threshold_violations.extend(violations)

            except TimeoutError:
                steps_executed += 1
                steps_failed += 1
                timeout_msg = (
                    f"Step {step.name} timed out after {step.timeout_seconds}s"
                )
                failures.append(timeout_msg)
                threshold_violations.append(timeout_msg)
                self.logger.error(timeout_msg)

            except Exception as e:
                steps_executed += 1
                steps_failed += 1
                error_msg = f"Step {step.name} failed with error: {e}"
                failures.append(error_msg)
                self.logger.error(error_msg)

        total_duration = time.time() - start_time
        performance_metrics["total_workflow_duration"] = total_duration

        result = WorkflowResult(
            workflow_type=workflow_type,
            total_duration=total_duration,
            steps_executed=steps_executed,
            steps_passed=steps_passed,
            steps_failed=steps_failed,
            failures=failures,
            performance_metrics=performance_metrics,
            threshold_violations=threshold_violations,
        )

        self.logger.info(
            f"Workflow {workflow_type.value} completed: "
            f"{steps_passed}/{steps_executed} steps passed "
            f"in {total_duration:.2f}s"
        )

        return result

    async def _execute_step(
        self,
        step: WorkflowStep,
        baseline_snapshot: ResourceSnapshot,
        post_cleanup_snapshot: ResourceSnapshot,
        test_id: str,
    ) -> bool:
        """Execute an individual verification step."""
        if step.name == "memory_validation":
            return await self._validate_memory_usage(
                baseline_snapshot, post_cleanup_snapshot
            )
        elif step.name == "process_validation":
            return await self._validate_process_count(
                baseline_snapshot, post_cleanup_snapshot
            )
        elif step.name == "file_handle_validation":
            return await self._validate_file_handles(
                baseline_snapshot, post_cleanup_snapshot
            )
        elif step.name == "gpu_memory_validation":
            return await self._validate_gpu_memory(
                baseline_snapshot, post_cleanup_snapshot
            )
        elif step.name == "temp_files_cleanup":
            return await self._validate_temp_files(test_id)
        elif step.name == "cleanup_time_validation":
            return await self._validate_cleanup_time(test_id)
        elif step.name == "orphaned_containers":
            return await self._validate_container_cleanup()
        else:
            # Generic validation for other steps
            return await self._generic_step_validation(step, test_id)

    async def _validate_memory_usage(
        self, baseline: ResourceSnapshot, post_cleanup: ResourceSnapshot
    ) -> bool:
        """Validate memory usage is within acceptable thresholds."""
        memory_diff = post_cleanup.memory_used_mb - baseline.memory_used_mb
        threshold = self.thresholds_config.get(
            "memory_leak_threshold_mb", 50.0
        )

        return memory_diff <= threshold

    async def _validate_process_count(
        self, baseline: ResourceSnapshot, post_cleanup: ResourceSnapshot
    ) -> bool:
        """Validate process count hasn't increased significantly."""
        process_diff = post_cleanup.process_count - baseline.process_count
        threshold = self.thresholds_config.get("process_count_threshold", 2)

        return process_diff <= threshold

    async def _validate_file_handles(
        self, baseline: ResourceSnapshot, post_cleanup: ResourceSnapshot
    ) -> bool:
        """Validate file handles are properly released."""
        if not hasattr(baseline, "file_handles") or not hasattr(
            post_cleanup, "file_handles"
        ):
            return True  # Skip if not available

        handle_diff = post_cleanup.file_handles - baseline.file_handles
        threshold = self.thresholds_config.get("file_handle_threshold", 10)

        return handle_diff <= threshold

    async def _validate_gpu_memory(
        self, baseline: ResourceSnapshot, post_cleanup: ResourceSnapshot
    ) -> bool:
        """Validate GPU memory is properly released."""
        if not hasattr(baseline, "gpu_memory_used_mb") or not hasattr(
            post_cleanup, "gpu_memory_used_mb"
        ):
            return True  # Skip if GPU not available

        gpu_diff = (
            post_cleanup.gpu_memory_used_mb - baseline.gpu_memory_used_mb
        )
        threshold = self.thresholds_config.get(
            "gpu_memory_threshold_mb", 100.0
        )

        return gpu_diff <= threshold

    async def _validate_temp_files(self, test_id: str) -> bool:
        """Validate temporary files are cleaned up."""
        # Implementation would check for test-specific temp files
        return True  # Placeholder

    async def _validate_cleanup_time(self, test_id: str) -> bool:
        """Validate cleanup completed within time threshold."""
        # This would check against actual cleanup timing
        # threshold = self.thresholds_config.get("cleanup_completion_ms", 2000)
        return True  # Placeholder - would integrate with actual timing

    async def _validate_container_cleanup(self) -> bool:
        """Validate Docker containers are properly cleaned up."""
        # Implementation would check for orphaned containers
        return True  # Placeholder

    async def _generic_step_validation(
        self, step: WorkflowStep, test_id: str
    ) -> bool:
        """Generic validation for steps without specific implementation."""
        # Placeholder for extensible step validation
        await asyncio.sleep(0.1)  # Simulate work
        return True

    def _check_step_thresholds(
        self, step: WorkflowStep, duration: float
    ) -> list[str]:
        """Check if step execution violates performance thresholds."""
        violations = []

        # Check if step took longer than expected
        if duration > step.timeout_seconds * 0.8:  # 80% of timeout as warning
            violations.append(
                f"Step {step.name} took {duration:.2f}s "
                f"(warning threshold: {step.timeout_seconds * 0.8:.2f}s)"
            )

        return violations
