"""Performance benchmark suite for comprehensive E2E testing.

This module defines the base infrastructure for performance benchmarks:
- BenchmarkConfig: Configuration for benchmark execution
- BenchmarkResult: Results container with metrics and validation
- BaseBenchmark: Abstract base for all benchmark implementations
- BenchmarkSuite: Main orchestrator for all performance tests

Specific benchmark types are implemented in separate modules:
- load_test.py: Normal operational load testing
- stress_test.py: Beyond-capacity stress testing
- endurance_test.py: Extended duration monitoring
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from tests.e2e.config.performance_thresholds import PerformanceThresholds

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""

    duration_seconds: float
    concurrent_users: int
    ramp_up_seconds: float
    iterations: int
    resource_monitoring_interval: float = 1.0

    def __post_init__(self) -> None:
        """Validate benchmark configuration."""
        if self.duration_seconds <= 0:
            raise ValueError("Duration must be positive")
        if self.concurrent_users <= 0:
            raise ValueError("Concurrent users must be positive")
        if self.iterations <= 0:
            raise ValueError("Iterations must be positive")


@dataclass
class BenchmarkResult:
    """Results from benchmark execution."""

    benchmark_name: str
    config: BenchmarkConfig
    start_time: float
    end_time: float
    success_count: int
    failure_count: int
    metrics: dict[str, Any]
    threshold_violations: list[str]

    @property
    def duration(self) -> float:
        """Total benchmark duration."""
        return self.end_time - self.start_time

    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        total = self.success_count + self.failure_count
        return (self.success_count / total * 100) if total > 0 else 0.0

    @property
    def throughput(self) -> float:
        """Operations per second."""
        total_ops = self.success_count + self.failure_count
        return total_ops / self.duration if self.duration > 0 else 0.0


class BaseBenchmark(ABC):
    """Base class for all performance benchmarks."""

    def __init__(self, name: str, thresholds: PerformanceThresholds) -> None:
        """Initialize benchmark."""
        self.name = name
        self.thresholds = thresholds
        self.logger = logging.getLogger(f"{__name__}.{name}")

    @abstractmethod
    async def execute(self, config: BenchmarkConfig) -> BenchmarkResult:
        """Execute the benchmark with given configuration."""
        pass

    @abstractmethod
    def validate_thresholds(self, result: BenchmarkResult) -> list[str]:
        """Validate results against performance thresholds."""
        pass

    def create_result(
        self,
        config: BenchmarkConfig,
        start_time: float,
        end_time: float,
        success_count: int,
        failure_count: int,
        metrics: dict[str, Any],
    ) -> BenchmarkResult:
        """Create benchmark result with threshold validation."""
        result = BenchmarkResult(
            benchmark_name=self.name,
            config=config,
            start_time=start_time,
            end_time=end_time,
            success_count=success_count,
            failure_count=failure_count,
            metrics=metrics,
            threshold_violations=[],
        )

        result.threshold_violations = self.validate_thresholds(result)
        return result


class BenchmarkSuite:
    """Main benchmark suite orchestrating all performance tests."""

    def __init__(self, thresholds: PerformanceThresholds) -> None:
        """Initialize benchmark suite with lazy loading of implementations."""
        self.thresholds = thresholds
        self.logger = logging.getLogger(__name__)
        self._benchmarks: dict[str, BaseBenchmark] = {}

    def _load_benchmarks(self) -> None:
        """Lazy load benchmark implementations to avoid circular imports."""
        if self._benchmarks:
            return

        # Import benchmark implementations
        from .endurance_test import EnduranceTestBenchmark
        from .load_test import LoadTestBenchmark
        from .stress_test import StressTestBenchmark

        self._benchmarks = {
            "load": LoadTestBenchmark(self.thresholds),
            "stress": StressTestBenchmark(self.thresholds),
            "endurance": EnduranceTestBenchmark(self.thresholds),
        }

    @property
    def benchmarks(self) -> dict[str, BaseBenchmark]:
        """Get available benchmarks with lazy loading."""
        self._load_benchmarks()
        return self._benchmarks

    async def run_all(
        self, configs: dict[str, BenchmarkConfig]
    ) -> dict[str, BenchmarkResult]:
        """Run all benchmarks with provided configurations."""
        results = {}

        for benchmark_name, benchmark in self.benchmarks.items():
            if benchmark_name in configs:
                self.logger.info(f"Starting {benchmark_name} benchmark")
                config = configs[benchmark_name]

                try:
                    result = await benchmark.execute(config)
                    results[benchmark_name] = result

                    self.logger.info(
                        f"{benchmark_name} completed: "
                        f"{result.success_rate:.1f}% success rate, "
                        f"{len(result.threshold_violations)} violations"
                    )
                except Exception as e:
                    self.logger.error(
                        f"{benchmark_name} benchmark failed: {e}"
                    )

        return results

    async def run_benchmark(
        self, benchmark_name: str, config: BenchmarkConfig
    ) -> BenchmarkResult:
        """Run a specific benchmark."""
        if benchmark_name not in self.benchmarks:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")

        benchmark = self.benchmarks[benchmark_name]
        return await benchmark.execute(config)

    def get_available_benchmarks(self) -> list[str]:
        """Get list of available benchmark names."""
        return list(self.benchmarks.keys())

    def validate_config(self, config: BenchmarkConfig) -> None:
        """Validate benchmark configuration against thresholds."""
        # Basic validation is done in BenchmarkConfig.__post_init__
        # Additional validations can be added here

        if config.duration_seconds > 3600:  # 1 hour max
            self.logger.warning(
                f"Long benchmark duration: {config.duration_seconds}s "
                "Consider breaking into smaller tests"
            )

        if config.concurrent_users > 100:
            self.logger.warning(
                f"High concurrent users: {config.concurrent_users} "
                "Ensure system can handle the load"
            )


def create_default_configs() -> dict[str, BenchmarkConfig]:
    """Create default benchmark configurations for quick testing."""
    return {
        "load": BenchmarkConfig(
            duration_seconds=60.0,
            concurrent_users=5,
            ramp_up_seconds=10.0,
            iterations=3,
            resource_monitoring_interval=1.0,
        ),
        "stress": BenchmarkConfig(
            duration_seconds=120.0,
            concurrent_users=20,
            ramp_up_seconds=15.0,
            iterations=2,
            resource_monitoring_interval=0.5,
        ),
        "endurance": BenchmarkConfig(
            duration_seconds=300.0,  # 5 minutes
            concurrent_users=3,
            ramp_up_seconds=5.0,
            iterations=1,
            resource_monitoring_interval=2.0,
        ),
    }
