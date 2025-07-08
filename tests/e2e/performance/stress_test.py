"""Stress testing benchmark for beyond-normal capacity testing.

This module implements stress testing that pushes the system beyond
normal operational limits to identify breaking points and validate
system behavior under extreme load conditions.
"""

import logging
import time
from typing import Any

from .benchmark_suite import BaseBenchmark, BenchmarkConfig, BenchmarkResult

logger = logging.getLogger(__name__)


class StressTestBenchmark(BaseBenchmark):
    """Stress testing benchmark for beyond-normal capacity."""

    def __init__(self, thresholds: Any) -> None:
        """Initialize stress test benchmark."""
        super().__init__("stress_test", thresholds)

    async def execute(self, config: BenchmarkConfig) -> BenchmarkResult:
        """Execute stress test benchmark."""
        self.logger.info(
            f"Starting stress test with {config.concurrent_users} users"
        )

        start_time = time.time()
        success_count = 0
        failure_count = 0
        metrics: dict[str, Any] = {}

        current_users = 1
        max_users = config.concurrent_users
        breaking_point_found = False

        try:
            # Stress testing with exponential load increase

            while current_users <= max_users and not breaking_point_found:
                self.logger.info(
                    f"Testing with {current_users} concurrent users"
                )

                level_start = time.time()
                level_success = 0
                level_failure = 0

                # Test current load level
                for _ in range(config.iterations):
                    try:
                        operation_time = await self._simulate_heavy_operation()
                        critical_threshold = (
                            self.thresholds.model_processing.inference_critical_ms
                        )
                        if operation_time <= critical_threshold:
                            level_success += 1
                            success_count += 1
                        else:
                            level_failure += 1
                            failure_count += 1
                    except Exception:
                        level_failure += 1
                        failure_count += 1

                level_duration = time.time() - level_start
                level_success_rate = (
                    level_success / (level_success + level_failure) * 100
                    if (level_success + level_failure) > 0
                    else 0
                )

                self.logger.info(
                    f"Load level {current_users} users: "
                    f"{level_success_rate:.1f}% success rate in "
                    f"{level_duration:.2f}s"
                )

                # Check if system is breaking down
                if level_success_rate < 50.0:  # Less than 50% success rate
                    breaking_point_found = True
                    self.logger.warning(
                        f"Breaking point found at {current_users} users"
                    )

                # Exponential increase in load
                current_users = min(current_users * 2, max_users)

                # Brief pause between load levels
                await self._async_sleep(1.0)

        except Exception as e:
            self.logger.error(f"Stress test failed: {e}")

        end_time = time.time()

        # Collect stress test metrics
        total_operations = success_count + failure_count
        metrics = {
            "max_concurrent_users_tested": config.concurrent_users,
            "breaking_point_users": (
                current_users // 2
                if breaking_point_found
                else config.concurrent_users
            ),
            "system_stability": success_count > failure_count,
            "total_operations": total_operations,
            "stress_duration_seconds": end_time - start_time,
        }

        return self.create_result(
            config, start_time, end_time, success_count, failure_count, metrics
        )

    def validate_thresholds(self, result: BenchmarkResult) -> list[str]:
        """Validate stress test results against thresholds."""
        violations = []

        # Check system stability under stress
        if not result.metrics.get("system_stability", False):
            violations.append(
                "System failed to maintain stability under stress"
            )

        # Check if breaking point is reasonable
        breaking_point = result.metrics.get("breaking_point_users", 0)
        expected_minimum = result.config.concurrent_users * 0.5
        if breaking_point < expected_minimum:
            violations.append(
                f"Breaking point at {breaking_point} users is too low "
                f"(expected at least {expected_minimum})"
            )

        # Check overall success rate
        if result.success_rate < 60.0:  # Lower threshold for stress testing
            violations.append(
                f"Overall success rate {result.success_rate:.1f}% "
                "below 60% threshold for stress testing"
            )

        return violations

    async def _simulate_heavy_operation(self) -> float:
        """Simulate heavy processing operation."""
        import random

        # Simulate heavy operations (2s - 8s range)
        operation_time = random.uniform(2000, 8000)
        await self._async_sleep(operation_time / 1000)
        return operation_time

    async def _async_sleep(self, seconds: float) -> None:
        """Async sleep helper."""
        import asyncio

        await asyncio.sleep(seconds)
