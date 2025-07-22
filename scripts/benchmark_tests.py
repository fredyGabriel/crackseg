#!/usr/bin/env python3
"""
Test performance benchmarking script. This script runs comprehensive
benchmarks to measure test execution performance improvements from
various optimization strategies. Part of subtask 7.5 - Test Execution
Performance Optimization. Usage: python scripts/benchmark_tests.py
[options]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import after path modification (required for module discovery)
from tests.utils.test_benchmark import (  # noqa: E402
    BenchmarkConfig,
    TestBenchmark,
)


def main() -> int:
    """
    Main entry point for benchmark script. Returns: Exit code (0 for
    success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Run test performance benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples: # Run benchmarks on GUI integration tests python
scripts/benchmark_tests.py tests/integration/gui/ # Run full benchmark
suite with detailed output python scripts/benchmark_tests.py --repeat
5 --verbose # Quick benchmark without memory monitoring python
scripts/benchmark_tests.py --quick tests/unit/ # Compare specific
optimizations python scripts/benchmark_tests.py --config
parallel_only,all_optimizations
""",
    )

    parser.add_argument(
        "test_patterns",
        nargs="*",
        default=["tests/integration/gui/"],
        help="Test patterns to benchmark (default: tests/integration/gui/)",
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Number of benchmark repetitions (default: 3)",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup runs (default: 1)",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick benchmark without memory monitoring",
    )

    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip baseline measurement",
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Comma-separated list of specific configurations to test",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test-artifacts/benchmarks"),
        help="Output directory for benchmark results",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Configure benchmark
    config = BenchmarkConfig(
        test_patterns=args.test_patterns,
        repeat_count=args.repeat,
        warmup_runs=args.warmup,
        include_baseline=not args.no_baseline,
        measure_memory=not args.quick,
        output_dir=args.output_dir,
    )

    if args.verbose:
        print("Benchmark configuration:")
        print(f"  Test patterns: {config.test_patterns}")
        print(f"  Repeat count: {config.repeat_count}")
        print(f"  Warmup runs: {config.warmup_runs}")
        print(f"  Memory monitoring: {config.measure_memory}")
        print(f"  Output directory: {config.output_dir}")
        print()

    # Run benchmark
    try:
        benchmark = TestBenchmark(config)

        if args.config:
            # Run specific configurations only
            specific_configs = args.config.split(",")
            print(f"Running benchmarks for configurations: {specific_configs}")
            results = {}

            all_configs = benchmark._get_benchmark_configurations()  # type: ignore
            for config_name in specific_configs:
                if config_name in all_configs:
                    optimization_config = all_configs[config_name]
                    result = benchmark._run_single_benchmark(
                        config_name, optimization_config
                    )
                    results[config_name] = result
                    benchmark.results.append(result)
                else:
                    print(f"Warning: Unknown configuration '{config_name}'")

            if results:
                benchmark._save_results()  # type: ignore
                benchmark._generate_report()  # type: ignore
        else:
            # Run full benchmark suite
            results = benchmark.run_benchmark_suite()

        # Display summary
        print("\nBenchmark Summary:")
        print("=" * 50)

        baseline_time = None
        for config_name, result in results.items():
            if config_name == "baseline":
                baseline_time = result.total_execution_time

            print(f"\n{config_name}:")
            print(f"  Execution time: {result.total_execution_time:.2f}s")
            tests_per_second = result.test_count / result.total_execution_time
            print(f"  Tests per second: {tests_per_second:.1f}")
            print(f"  Memory peak: {result.memory_usage_peak_mb:.1f}MB")
            print(f"  Workers: {result.worker_count}")

            if baseline_time and config_name != "baseline":
                speedup = baseline_time / result.total_execution_time
                print(f"  Speedup: {speedup:.2f}x")

        if args.verbose:
            print(f"\nDetailed results saved to: {config.output_dir}")

        return 0

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"Benchmark failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
