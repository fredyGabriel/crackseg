"""
Test benchmarking and performance measurement tools. This module
provides benchmarking capabilities to measure test execution
performance improvements from the optimization framework. Part of
subtask 7.5 - Test Execution Performance Optimization.
"""

import json
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .performance_optimizer import OptimizationConfig


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    configuration_name: str
    total_execution_time: float
    average_test_time: float
    setup_overhead: float
    parallel_efficiency: float
    fixture_cache_hit_rate: float
    memory_usage_peak_mb: float
    test_count: int
    failed_tests: int
    worker_count: int
    optimization_settings: dict[str, Any]


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""

    test_patterns: list[str]
    repeat_count: int = 3
    warmup_runs: int = 1
    include_baseline: bool = True
    measure_memory: bool = True
    output_dir: Path = Path("test-artifacts/benchmarks")


class TestBenchmark:
    """Test execution benchmark and performance measurement."""

    def __init__(self, config: BenchmarkConfig) -> None:
        """Initialize test benchmark.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[BenchmarkResult] = []

    def run_benchmark_suite(self) -> dict[str, BenchmarkResult]:
        """Run complete benchmark suite with different configurations.

        Returns:
            Dictionary of configuration name to benchmark results
        """
        configurations = self._generate_test_configurations()
        results = {}

        for config_name, opt_config in configurations.items():
            print(f"Running benchmark: {config_name}")
            result = self._run_single_benchmark(config_name, opt_config)
            results[config_name] = result
            self.results.append(result)

        self._save_benchmark_results(results)
        return results

    def _generate_test_configurations(self) -> dict[str, OptimizationConfig]:
        """Generate different optimization configurations to test.

        Returns:
            Dictionary of configuration name to OptimizationConfig
        """
        configs = {}

        # Baseline configuration (no optimizations)
        if self.config.include_baseline:
            configs["baseline"] = OptimizationConfig(
                enable_fixture_caching=False,
                enable_selective_running=False,
                auto_parallel_detection=False,
                optimal_worker_count=1,
            )

        # Fixture caching only
        configs["fixture_cache"] = OptimizationConfig(
            enable_fixture_caching=True,
            enable_selective_running=False,
            auto_parallel_detection=False,
            optimal_worker_count=1,
        )

        # Parallel execution only
        configs["parallel"] = OptimizationConfig(
            enable_fixture_caching=False,
            enable_selective_running=False,
            auto_parallel_detection=True,
            optimal_worker_count=-1,
        )

        # Full optimization
        configs["full_optimization"] = OptimizationConfig(
            enable_fixture_caching=True,
            enable_selective_running=True,
            auto_parallel_detection=True,
            optimal_worker_count=-1,
        )

        return configs

    def _run_single_benchmark(
        self, config_name: str, opt_config: OptimizationConfig
    ) -> BenchmarkResult:
        """Run benchmark for a single configuration.

        Args:
            config_name: Name of the configuration
            opt_config: Optimization configuration

        Returns:
            Benchmark result
        """
        execution_times = []
        setup_times = []

        # Warmup runs
        for _ in range(self.config.warmup_runs):
            self._execute_tests(opt_config, warmup=True)

        # Actual benchmark runs
        for _run_idx in range(self.config.repeat_count):
            start_time = time.time()

            # Measure setup overhead
            setup_start = time.time()
            test_result = self._execute_tests(opt_config)
            setup_end = time.time()

            end_time = time.time()

            execution_time = end_time - start_time
            setup_time = setup_end - setup_start

            execution_times.append(execution_time)
            setup_times.append(setup_time)

        # Calculate metrics
        avg_execution_time = statistics.mean(execution_times)
        avg_setup_time = statistics.mean(setup_times)

        # Parse test results (placeholder implementation)
        test_count = test_result.get("test_count", 0)
        failed_tests = test_result.get("failed_tests", 0)

        # Calculate parallel efficiency (simplified)
        worker_count = (
            opt_config.optimal_worker_count
            if opt_config.optimal_worker_count > 0
            else 1
        )
        parallel_efficiency = 1.0 / worker_count if worker_count > 1 else 1.0

        return BenchmarkResult(
            configuration_name=config_name,
            total_execution_time=avg_execution_time,
            average_test_time=avg_execution_time / max(test_count, 1),
            setup_overhead=avg_setup_time,
            parallel_efficiency=parallel_efficiency,
            # Would be calculated from actual cache metrics
            fixture_cache_hit_rate=0.0,
            memory_usage_peak_mb=0.0,  # Would be measured during execution
            test_count=test_count,
            failed_tests=failed_tests,
            worker_count=worker_count,
            optimization_settings=asdict(opt_config),
        )

    def _execute_tests(
        self, opt_config: OptimizationConfig, warmup: bool = False
    ) -> dict[str, Any]:
        """Execute tests with given optimization configuration.

        Args:
            opt_config: Optimization configuration
            warmup: Whether this is a warmup run

        Returns:
            Test execution results
        """
        # Build pytest command
        cmd = ["python", "-m", "pytest"]

        # Add test patterns
        cmd.extend(self.config.test_patterns)

        # Add optimization flags
        if (
            opt_config.auto_parallel_detection
            and opt_config.optimal_worker_count != 1
        ):
            worker_count = (
                opt_config.optimal_worker_count
                if opt_config.optimal_worker_count > 0
                else 2
            )
            cmd.extend(["-n", str(worker_count)])

        # Add output options
        if warmup:
            cmd.extend(["-q", "--tb=no"])
        else:
            cmd.extend(["--tb=short"])

        try:
            # Execute tests
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            # Parse results (simplified)
            output_lines = result.stdout.split("\n")
            test_count = 0
            failed_tests = 0

            for line in output_lines:
                if "passed" in line or "failed" in line:
                    # Extract test counts from pytest output
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "passed":
                            test_count += int(parts[i - 1]) if i > 0 else 0
                        elif part == "failed":
                            failed_tests += int(parts[i - 1]) if i > 0 else 0

            return {
                "test_count": test_count,
                "failed_tests": failed_tests,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

        except subprocess.TimeoutExpired:
            return {
                "test_count": 0,
                "failed_tests": 0,
                "return_code": -1,
                "error": "Test execution timeout",
            }
        except Exception as e:
            return {
                "test_count": 0,
                "failed_tests": 0,
                "return_code": -1,
                "error": str(e),
            }

    def _save_benchmark_results(
        self, results: dict[str, BenchmarkResult]
    ) -> None:
        """Save benchmark results to file.

        Args:
            results: Dictionary of benchmark results
        """
        output_file = (
            self.config.output_dir / f"benchmark_{int(time.time())}.json"
        )

        # Convert results to JSON-serializable format
        serializable_results = {}
        for config_name, result in results.items():
            serializable_results[config_name] = asdict(result)

        # Add metadata
        benchmark_data = {
            "timestamp": time.time(),
            "config": asdict(self.config),
            "results": serializable_results,
            "summary": self._generate_summary(results),
        }

        try:
            with output_file.open("w") as f:
                json.dump(benchmark_data, f, indent=2, default=str)
            print(f"Benchmark results saved to: {output_file}")
        except Exception as e:
            print(f"Failed to save benchmark results: {e}")

    def _generate_summary(
        self, results: dict[str, BenchmarkResult]
    ) -> dict[str, Any]:
        """Generate summary statistics from benchmark results.

        Args:
            results: Dictionary of benchmark results

        Returns:
            Summary statistics
        """
        if not results:
            return {}

        baseline_result = results.get("baseline")
        summary = {
            "total_configurations": len(results),
            "performance_improvements": {},
        }

        if baseline_result:
            baseline_time = baseline_result.total_execution_time

            for config_name, result in results.items():
                if config_name != "baseline":
                    improvement = (
                        (baseline_time - result.total_execution_time)
                        / baseline_time
                        * 100
                    )
                    summary["performance_improvements"][
                        config_name
                    ] = improvement

        return summary


def run_performance_benchmarks(
    test_patterns: list[str] | None = None, output_dir: Path | None = None
) -> dict[str, BenchmarkResult]:
    """Convenience function to run performance benchmarks.

    Args:
        test_patterns: Test patterns to benchmark
        output_dir: Output directory for results

    Returns:
        Dictionary of benchmark results
    """
    patterns = test_patterns or ["tests/unit/", "tests/integration/"]
    out_dir = output_dir or Path("test-artifacts/benchmarks")

    config = BenchmarkConfig(
        test_patterns=patterns,
        repeat_count=3,
        warmup_runs=1,
        include_baseline=True,
        output_dir=out_dir,
    )

    benchmark = TestBenchmark(config)
    return benchmark.run_benchmark_suite()
