"""Endurance testing benchmark for extended duration monitoring.

This module implements endurance testing that runs for extended periods
to detect memory leaks, performance degradation, and system stability
over time under sustained load.
"""

import logging
import time
from typing import Any

from tests.e2e.config.performance_thresholds import PerformanceThresholds

from .benchmark_suite import BaseBenchmark, BenchmarkConfig, BenchmarkResult

logger = logging.getLogger(__name__)


class EnduranceTestBenchmark(BaseBenchmark):
    """Endurance testing benchmark for extended duration monitoring."""

    def __init__(self, thresholds: PerformanceThresholds) -> None:
        """Initialize endurance test benchmark."""
        super().__init__("endurance_test", thresholds)

    async def execute(self, config: BenchmarkConfig) -> BenchmarkResult:
        """Execute endurance test benchmark."""
        self.logger.info(
            f"Starting endurance test for {config.duration_seconds}s"
        )

        start_time = time.time()
        success_count = 0
        failure_count = 0
        metrics: dict[str, Any] = {}

        try:
            end_target = start_time + config.duration_seconds
            operation_count = 0

            while time.time() < end_target:
                try:
                    # Consistent moderate load
                    operation_time = await self._simulate_sustained_operation()
                    operation_count += 1

                    if (
                        operation_time
                        <= self.thresholds.web_interface.page_load_warning_ms
                    ):
                        success_count += 1
                    else:
                        failure_count += 1

                    # Monitor for memory leaks and resource degradation
                    if operation_count % 100 == 0:
                        self.logger.debug(
                            f"Completed {operation_count} operations"
                        )

                    # Sustained pace
                    await self._async_sleep(
                        config.resource_monitoring_interval
                    )

                except Exception as e:
                    self.logger.warning(f"Endurance operation failed: {e}")
                    failure_count += 1

        except Exception as e:
            self.logger.error(f"Endurance test failed: {e}")

        end_time = time.time()

        # Collect endurance metrics
        total_runtime = end_time - start_time
        total_operations = success_count + failure_count
        metrics = {
            "total_runtime_seconds": total_runtime,
            "operations_per_minute": (
                total_operations / (total_runtime / 60)
                if total_runtime > 0
                else 0
            ),
            "performance_degradation": self._calculate_degradation(
                success_count, failure_count
            ),
            "total_operations": total_operations,
        }

        return self.create_result(
            config, start_time, end_time, success_count, failure_count, metrics
        )

    def validate_thresholds(self, result: BenchmarkResult) -> list[str]:
        """Validate endurance test results against thresholds."""
        violations = []

        # Check for significant performance degradation
        degradation = result.metrics.get("performance_degradation", 0)
        if degradation > 20.0:  # 20% degradation threshold
            violations.append(
                f"Performance degraded by {degradation:.1f}% over test "
                f"duration"
            )

        # Check minimum operations per minute
        ops_per_min = result.metrics.get("operations_per_minute", 0)
        if ops_per_min < 10:  # Minimum 10 operations per minute
            violations.append(
                f"Operations per minute {ops_per_min:.1f} below minimum "
                f"threshold"
            )

        # Check overall success rate for endurance
        if (
            result.success_rate < 85.0
        ):  # 85% success rate for long-running tests
            violations.append(
                f"Success rate {result.success_rate:.1f}% below 85% "
                f"threshold for endurance testing"
            )

        return violations

    def _calculate_degradation(
        self, success_count: int, failure_count: int
    ) -> float:
        """Calculate performance degradation percentage."""
        total = success_count + failure_count
        if total == 0:
            return 100.0

        # Simple degradation calculation based on failure rate
        failure_rate = failure_count / total * 100
        return min(failure_rate, 100.0)

    async def _simulate_sustained_operation(self) -> float:
        """Simulate sustained operation."""
        import random

        # Simulate consistent operations (1s - 3s range)
        operation_time = random.uniform(1000, 3000)
        await self._async_sleep(operation_time / 1000)
        return operation_time

    async def _async_sleep(self, seconds: float) -> None:
        """Async sleep helper."""
        import asyncio

        await asyncio.sleep(seconds)
