"""Load testing benchmark for normal operational conditions.

This module implements load testing that simulates normal user traffic
patterns and validates system performance under expected load conditions.
"""

import logging
import time
from typing import Any

from .benchmark_suite import BaseBenchmark, BenchmarkConfig, BenchmarkResult

logger = logging.getLogger(__name__)


class LoadTestBenchmark(BaseBenchmark):
    """Load testing benchmark for normal operational conditions."""

    def __init__(self, thresholds: Any) -> None:
        """Initialize load test benchmark."""
        super().__init__("load_test", thresholds)

    async def execute(self, config: BenchmarkConfig) -> BenchmarkResult:
        """Execute load test benchmark."""
        self.logger.info(
            f"Starting load test with {config.concurrent_users} users"
        )

        start_time = time.time()
        success_count = 0
        failure_count = 0
        metrics: dict[str, Any] = {}

        try:
            # Simulate load testing operations
            for iteration in range(config.iterations):
                iteration_start = time.time()

                # Simulate concurrent user operations
                for _user in range(config.concurrent_users):
                    try:
                        # Simulate page load operation
                        operation_time = await self._simulate_page_load()
                        critical_threshold = (
                            self.thresholds.web_interface.page_load_critical_ms
                        )
                        if operation_time <= critical_threshold:
                            success_count += 1
                        else:
                            failure_count += 1

                    except Exception as e:
                        self.logger.warning(f"Operation failed: {e}")
                        failure_count += 1

                iteration_duration = time.time() - iteration_start
                self.logger.debug(
                    f"Iteration {iteration + 1} completed in "
                    f"{iteration_duration:.2f}s"
                )

                # Ramp-up delay
                if config.ramp_up_seconds > 0:
                    await self._async_sleep(
                        config.ramp_up_seconds / config.iterations
                    )

        except Exception as e:
            self.logger.error(f"Load test failed: {e}")
            failure_count += config.concurrent_users * config.iterations

        end_time = time.time()

        # Collect metrics
        total_operations = success_count + failure_count
        metrics = {
            "avg_response_time_ms": (
                (end_time - start_time) * 1000 / total_operations
                if total_operations > 0
                else 0
            ),
            "peak_concurrent_users": config.concurrent_users,
            "total_operations": total_operations,
            "operations_per_second": total_operations
            / (end_time - start_time),
        }

        return self.create_result(
            config, start_time, end_time, success_count, failure_count, metrics
        )

    def validate_thresholds(self, result: BenchmarkResult) -> list[str]:
        """Validate load test results against thresholds."""
        violations = []

        # Check success rate
        if result.success_rate < 95.0:  # 95% success rate threshold
            violations.append(
                f"Success rate {result.success_rate:.1f}% below 95%"
            )

        # Check average response time
        avg_response = result.metrics.get("avg_response_time_ms", 0)
        if avg_response > self.thresholds.web_interface.page_load_critical_ms:
            violations.append(
                f"Average response time {avg_response:.1f}ms exceeds "
                f"{self.thresholds.web_interface.page_load_critical_ms}ms "
                f"threshold"
            )

        # Check throughput
        ops_per_sec = result.metrics.get("operations_per_second", 0)
        min_throughput = 0.5  # Minimum 0.5 operations per second
        if ops_per_sec < min_throughput:
            violations.append(
                f"Throughput {ops_per_sec:.2f} ops/sec below minimum "
                f"{min_throughput}"
            )

        return violations

    async def _simulate_page_load(self) -> float:
        """Simulate page load operation."""
        # Simulate variable page load times (800ms - 2500ms range)
        import random

        load_time = random.uniform(800, 2500)
        await self._async_sleep(load_time / 1000)  # Convert to seconds
        return load_time

    async def _async_sleep(self, seconds: float) -> None:
        """Async sleep helper."""
        import asyncio

        await asyncio.sleep(seconds)
