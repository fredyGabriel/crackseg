"""Test benchmarking and performance measurement tools.

This module provides benchmarking capabilities to measure test execution
performance improvements from the optimization framework.
Part of subtask 7.5 - Test Execution Performance Optimization.
"""

import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

from crackseg.dataclasses import asdict, dataclass

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
        configurations = self._get_benchmark_configurations()
        results: dict[str, BenchmarkResult] = {}

        for config_name, optimization_config in configurations.items():
            print(f"Running benchmark: {config_name}")
            result = self._run_single_benchmark(
                config_name, optimization_config
            )
            results[config_name] = result
            self.results.append(result)

        self._save_results()
        self._generate_report()
        return results

    def _get_benchmark_configurations(self) -> dict[str, OptimizationConfig]:
        """Get different optimization configurations for benchmarking.

        Returns:
            Dictionary of configuration name to optimization config
        """
        return {
            "baseline": OptimizationConfig(
                enable_fixture_caching=False,
                enable_selective_running=False,
                auto_parallel_detection=False,
                optimal_worker_count=1,
                metrics_collection_enabled=False,
            ),
            "fixture_cache_only": OptimizationConfig(
                enable_fixture_caching=True,
                enable_selective_running=False,
                auto_parallel_detection=False,
                optimal_worker_count=1,
            ),
            "parallel_only": OptimizationConfig(
                enable_fixture_caching=False,
                enable_selective_running=False,
                auto_parallel_detection=True,
                optimal_worker_count=-1,
            ),
            "selective_only": OptimizationConfig(
                enable_fixture_caching=False,
                enable_selective_running=True,
                auto_parallel_detection=False,
                optimal_worker_count=1,
            ),
            "all_optimizations": OptimizationConfig(
                enable_fixture_caching=True,
                enable_selective_running=True,
                auto_parallel_detection=True,
                optimal_worker_count=-1,
            ),
        }

    def _run_single_benchmark(
        self, config_name: str, optimization_config: OptimizationConfig
    ) -> BenchmarkResult:
        """Run benchmark for a single configuration.

        Args:
            config_name: Name of the configuration
            optimization_config: Optimization configuration

        Returns:
            Benchmark result
        """
        execution_times: list[float] = []
        setup_times: list[float] = []
        memory_usage: list[float] = []
        latest_result: dict[str, Any] = {}

        # Warmup runs
        for _ in range(self.config.warmup_runs):
            self._execute_tests(optimization_config, measure=False)

        # Actual benchmark runs
        for run_idx in range(self.config.repeat_count):
            print(f"  Run {run_idx + 1}/{self.config.repeat_count}")
            latest_result = self._execute_tests(
                optimization_config, measure=True
            )

            execution_times.append(float(latest_result["execution_time"]))
            setup_times.append(float(latest_result["setup_time"]))
            if self.config.measure_memory:
                memory_usage.append(float(latest_result["memory_usage"]))

        # Calculate statistics
        avg_execution_time: float = statistics.mean(execution_times)
        avg_setup_time: float = statistics.mean(setup_times)
        avg_memory: float = (
            statistics.mean(memory_usage) if memory_usage else 0.0
        )

        # Calculate efficiency metrics
        parallel_efficiency = self._calculate_parallel_efficiency(
            optimization_config, avg_execution_time
        )

        return BenchmarkResult(
            configuration_name=config_name,
            total_execution_time=avg_execution_time,
            average_test_time=avg_execution_time
            / max(1, int(latest_result["test_count"])),
            setup_overhead=avg_setup_time,
            parallel_efficiency=parallel_efficiency,
            fixture_cache_hit_rate=float(
                latest_result.get("cache_hit_rate", 0.0)
            ),
            memory_usage_peak_mb=avg_memory,
            test_count=int(latest_result["test_count"]),
            failed_tests=int(latest_result["failed_tests"]),
            worker_count=int(latest_result["worker_count"]),
            optimization_settings=asdict(optimization_config),
        )

    def _execute_tests(
        self, config: OptimizationConfig, measure: bool = True
    ) -> dict[str, Any]:
        """Execute tests with given configuration.

        Args:
            config: Optimization configuration
            measure: Whether to collect detailed measurements

        Returns:
            Execution results
        """
        from .performance_optimizer import TestPerformanceOptimizer

        optimizer = TestPerformanceOptimizer(config)

        # Build pytest command
        base_args = ["pytest"] + self.config.test_patterns
        if measure:
            base_args.extend(["--durations=0", "-v"])
        else:
            base_args.extend(["--quiet"])

        optimized_args = optimizer.get_optimized_pytest_command(base_args)

        # Execute and measure
        start_time = time.time()
        if measure and self.config.measure_memory:
            result = self._execute_with_memory_monitoring(optimized_args)
        else:
            result = self._execute_simple(optimized_args)

        execution_time = time.time() - start_time

        return {
            "execution_time": execution_time,
            "setup_time": result.get("setup_time", 0.0),
            "test_count": result.get("test_count", 0),
            "failed_tests": result.get("failed_tests", 0),
            "worker_count": self._extract_worker_count(optimized_args),
            "cache_hit_rate": result.get("cache_hit_rate", 0.0),
            "memory_usage": result.get("memory_usage", 0.0),
        }

    def _execute_simple(self, args: list[str]) -> dict[str, Any]:
        """Execute pytest with simple output parsing.

        Args:
            args: Pytest arguments

        Returns:
            Execution results
        """
        try:
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            # Parse basic metrics from output
            output = result.stdout + result.stderr
            test_count = output.count(" PASSED") + output.count(" FAILED")
            failed_tests = output.count(" FAILED")

            return {
                "test_count": test_count,
                "failed_tests": failed_tests,
                "setup_time": 0.0,  # Would need more sophisticated parsing
                "cache_hit_rate": 0.0,
                "memory_usage": 0.0,
            }

        except subprocess.TimeoutExpired:
            return {
                "test_count": 0,
                "failed_tests": 0,
                "setup_time": 0.0,
                "cache_hit_rate": 0.0,
                "memory_usage": 0.0,
            }

    def _execute_with_memory_monitoring(
        self, args: list[str]
    ) -> dict[str, Any]:
        """Execute pytest with memory monitoring.

        Args:
            args: Pytest arguments

        Returns:
            Execution results with memory usage
        """
        import psutil

        # Start pytest process
        process = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # Monitor memory usage
        memory_samples: list[float] = []
        try:
            ps_process = psutil.Process(process.pid)
            while process.poll() is None:
                try:
                    memory_info = ps_process.memory_info()
                    memory_samples.append(
                        float(memory_info.rss / (1024 * 1024))
                    )  # MB
                except psutil.NoSuchProcess:
                    break
                time.sleep(0.1)  # Sample every 100ms

            stdout, stderr = process.communicate(timeout=300)
            output = stdout + stderr

        except (subprocess.TimeoutExpired, psutil.NoSuchProcess):
            process.kill()
            output = ""
            memory_samples = [0.0]

        # Parse results
        test_count = output.count(" PASSED") + output.count(" FAILED")
        failed_tests = output.count(" FAILED")
        peak_memory = max(memory_samples) if memory_samples else 0.0

        return {
            "test_count": test_count,
            "failed_tests": failed_tests,
            "setup_time": 0.0,
            "cache_hit_rate": 0.0,
            "memory_usage": peak_memory,
        }

    def _extract_worker_count(self, args: list[str]) -> int:
        """Extract worker count from pytest arguments.

        Args:
            args: Pytest arguments

        Returns:
            Number of workers
        """
        for i, arg in enumerate(args):
            if arg == "-n" and i + 1 < len(args):
                try:
                    return int(args[i + 1])
                except ValueError:
                    return 1
        return 1

    def _calculate_parallel_efficiency(
        self, config: OptimizationConfig, execution_time: float
    ) -> float:
        """Calculate parallel execution efficiency.

        Args:
            config: Optimization configuration
            execution_time: Total execution time

        Returns:
            Parallel efficiency ratio (0.0 to 1.0)
        """
        # This is a simplified calculation
        # In practice, you'd compare against sequential baseline
        worker_count = config.optimal_worker_count
        if worker_count <= 1:
            return 1.0

        # Theoretical maximum speedup is worker_count
        # Real efficiency is usually 60-80% due to overhead
        estimated_sequential_time = execution_time * worker_count
        theoretical_best = estimated_sequential_time / worker_count
        efficiency = theoretical_best / execution_time

        return min(efficiency, 1.0)

    def _save_results(self) -> None:
        """Save benchmark results to disk."""
        timestamp = int(time.time())
        results_file = self.config.output_dir / f"benchmark_{timestamp}.json"

        # Convert Path objects to strings for JSON serialization
        config_dict = asdict(self.config)
        config_dict["output_dir"] = str(config_dict["output_dir"])

        results_data = {
            "timestamp": timestamp,
            "config": config_dict,
            "results": [asdict(result) for result in self.results],
        }

        with results_file.open("w") as f:
            json.dump(results_data, f, indent=2)

        print(f"Benchmark results saved to: {results_file}")

    def _generate_report(self) -> None:
        """Generate a human-readable benchmark report."""
        report_file = self.config.output_dir / "latest_benchmark_report.txt"

        with report_file.open("w") as f:
            f.write("Test Performance Benchmark Report\n")
            f.write("=" * 40 + "\n\n")

            # Find baseline for comparison
            baseline = next(
                (
                    r
                    for r in self.results
                    if r.configuration_name == "baseline"
                ),
                None,
            )

            for result in self.results:
                f.write(f"Configuration: {result.configuration_name}\n")
                f.write("-" * 30 + "\n")
                f.write(
                    f"Total execution time: "
                    f"{result.total_execution_time:.2f}s\n"
                )
                f.write(
                    f"Average test time: {result.average_test_time:.3f}s\n"
                )
                f.write(
                    f"Parallel efficiency: {result.parallel_efficiency:.2%}\n"
                )
                f.write(
                    f"Cache hit rate: {result.fixture_cache_hit_rate:.2%}\n"
                )
                f.write(f"Memory usage: {result.memory_usage_peak_mb:.1f}MB\n")
                f.write(f"Worker count: {result.worker_count}\n")

                if baseline and result != baseline:
                    speedup = (
                        baseline.total_execution_time
                        / result.total_execution_time
                    )
                    f.write(f"Speedup vs baseline: {speedup:.2f}x\n")

                f.write("\n")

        print(f"Benchmark report saved to: {report_file}")
