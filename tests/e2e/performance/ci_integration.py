"""CI/CD integration for performance benchmarking.

This module provides integration utilities for running performance benchmarks
in CI/CD environments, handling configuration, result processing, and
coordination with cleanup validation systems.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import UTC
from pathlib import Path
from typing import Any

from tests.e2e.config.performance_thresholds import PerformanceThresholds
from tests.e2e.performance.benchmark_runner import BenchmarkRunner
from tests.e2e.performance.benchmark_suite import BenchmarkConfig

logger = logging.getLogger(__name__)


class PerformanceCIIntegration:
    """Integration layer for performance testing in CI/CD environments."""

    def __init__(
        self,
        results_dir: Path | str = "performance-results",
        artifacts_dir: Path | str = "benchmark-artifacts",
        thresholds_config: (
            Path | str
        ) = "configs/testing/performance_thresholds.yaml",
    ) -> None:
        """Initialize CI integration."""
        self.results_dir = Path(results_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.thresholds_config = Path(thresholds_config)

        # Ensure directories exist
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Load performance thresholds
        self.thresholds = PerformanceThresholds.from_config(
            self.thresholds_config
        )

        # Initialize benchmark runner
        self.benchmark_runner = BenchmarkRunner(
            thresholds=self.thresholds,
            enable_resource_monitoring=self._get_env_bool(
                "RESOURCE_MONITORING_ENABLED", True
            ),
            monitoring_interval=float(
                os.getenv("PERFORMANCE_MONITORING_INTERVAL", "1.0")
            ),
        )

        self.logger = logging.getLogger(__name__)

    def _get_env_bool(self, env_var: str, default: bool) -> bool:
        """Get boolean environment variable."""
        value = os.getenv(env_var, str(default)).lower()
        return value in ("true", "1", "yes", "on")

    async def run_benchmark_suite(
        self, suite_name: str = "smoke"
    ) -> dict[str, Any]:
        """Run performance benchmark suite for CI/CD."""
        self.logger.info(f"Running {suite_name} benchmark suite for CI/CD")

        # Get benchmark configurations based on suite name
        configs = self._get_benchmark_configs(suite_name)

        # Run benchmarks with monitoring
        results = await self.benchmark_runner.run_benchmark_suite(configs)

        # Process and save results for CI
        processed_results = self._process_results_for_ci(results, suite_name)

        # Save results
        await self._save_ci_results(processed_results, suite_name)

        return processed_results

    def _get_benchmark_configs(
        self, suite_name: str
    ) -> dict[str, BenchmarkConfig]:
        """Get benchmark configurations based on suite name."""
        # Base configuration
        base_config = BenchmarkConfig(
            duration_seconds=60.0,
            iterations=10,
            ramp_up_seconds=2.0,
            concurrent_users=1,
            resource_monitoring_interval=1.0,
        )

        configs = {}

        if suite_name in ("smoke", "full"):
            configs["smoke_test"] = BenchmarkConfig(
                duration_seconds=30.0,
                iterations=5,
                ramp_up_seconds=1.0,
                concurrent_users=1,
                resource_monitoring_interval=1.0,
            )

        if suite_name in ("stress", "full"):
            configs["stress_test"] = BenchmarkConfig(
                duration_seconds=120.0,
                iterations=20,
                ramp_up_seconds=3.0,
                concurrent_users=5,
                resource_monitoring_interval=1.0,
            )

        if suite_name in ("load", "full"):
            configs["load_test"] = BenchmarkConfig(
                duration_seconds=180.0,
                iterations=30,
                ramp_up_seconds=5.0,
                concurrent_users=10,
                resource_monitoring_interval=1.0,
            )

        if suite_name in ("endurance", "full"):
            configs["endurance_test"] = BenchmarkConfig(
                duration_seconds=600.0,  # 10 minutes
                iterations=50,
                ramp_up_seconds=5.0,
                concurrent_users=3,
                resource_monitoring_interval=1.0,
            )

        if suite_name == "regression":
            # Regression tests focus on performance stability
            configs["regression_test"] = BenchmarkConfig(
                duration_seconds=90.0,
                iterations=15,
                ramp_up_seconds=3.0,
                concurrent_users=2,
                resource_monitoring_interval=1.0,
            )

        if not configs:
            # Default to smoke test if unknown suite
            configs["default_test"] = base_config

        return configs

    def _process_results_for_ci(
        self, results: dict[str, dict[str, Any]], suite_name: str
    ) -> dict[str, Any]:
        """Process benchmark results for CI/CD consumption."""
        processed: dict[str, Any] = {
            "suite_name": suite_name,
            "ci_metadata": {
                "build_number": os.getenv("GITHUB_RUN_NUMBER", "local"),
                "commit_sha": os.getenv("GITHUB_SHA", "unknown"),
                "branch": os.getenv("GITHUB_REF_NAME", "unknown"),
                "workflow": "performance-ci",
                "timestamp": self._get_timestamp(),
            },
            "performance_summary": {
                "total_benchmarks": len(results),
                "successful_benchmarks": 0,
                "failed_benchmarks": 0,
                "total_violations": 0,
            },
            "benchmark_results": {},
            "threshold_violations": [],
            "performance_metrics": {},
        }

        # Process each benchmark result
        for benchmark_name, result in results.items():
            if "error" in result:
                summary = processed["performance_summary"]
                if isinstance(summary, dict):
                    summary["failed_benchmarks"] = (
                        summary.get("failed_benchmarks", 0) + 1
                    )
                processed["benchmark_results"][benchmark_name] = {
                    "status": "failed",
                    "error": result["error"],
                }
                continue

            # Process successful benchmark
            summary = processed["performance_summary"]
            if isinstance(summary, dict):
                summary["successful_benchmarks"] = (
                    summary.get("successful_benchmarks", 0) + 1
                )

            # Extract benchmark data
            benchmark_data = result.get("benchmark_result", {})
            resource_data = result.get("resource_metrics", {})

            # Check for violations
            violations = benchmark_data.get("threshold_violations", [])
            threshold_violations = processed["threshold_violations"]
            if isinstance(threshold_violations, list):
                threshold_violations.extend(violations)

            summary = processed["performance_summary"]
            if isinstance(summary, dict):
                summary["total_violations"] = summary.get(
                    "total_violations", 0
                ) + len(violations)

            # Store processed benchmark result
            processed["benchmark_results"][benchmark_name] = {
                "status": "success",
                "success_rate": benchmark_data.get("success_rate", 0.0),
                "throughput": benchmark_data.get("throughput", 0.0),
                "duration": benchmark_data.get("duration", 0.0),
                "violations_count": len(violations),
                "violations": violations[:5],  # First 5 violations
                "resource_metrics": self._summarize_resource_metrics(
                    resource_data
                ),
            }

            # Store detailed metrics
            if benchmark_data.get("metrics"):
                processed["performance_metrics"][benchmark_name] = (
                    benchmark_data["metrics"]
                )

        # Add overall assessment
        processed["overall_status"] = self._assess_overall_status(processed)

        return processed

    def _summarize_resource_metrics(
        self, resource_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Summarize resource metrics for CI reporting."""
        if not resource_data:
            return {}

        return {
            "peak_cpu_percent": resource_data.get("peak_cpu_percent", 0.0),
            "peak_memory_mb": resource_data.get("peak_memory_mb", 0.0),
            "peak_gpu_memory_mb": resource_data.get("peak_gpu_memory_mb", 0.0),
            "peak_gpu_utilization": resource_data.get(
                "peak_gpu_utilization", 0.0
            ),
            "avg_cpu_percent": resource_data.get("avg_cpu_percent", 0.0),
            "avg_memory_mb": resource_data.get("avg_memory_mb", 0.0),
            "total_disk_read_mb": resource_data.get("total_disk_read_mb", 0.0),
            "total_disk_write_mb": resource_data.get(
                "total_disk_write_mb", 0.0
            ),
        }

    def _assess_overall_status(self, processed: dict[str, Any]) -> str:
        """Assess overall benchmark suite status."""
        summary = processed["performance_summary"]

        if not isinstance(summary, dict):
            return "ERROR"

        failed_benchmarks = summary.get("failed_benchmarks", 0)
        total_violations = summary.get("total_violations", 0)
        successful_benchmarks = summary.get("successful_benchmarks", 0)

        if failed_benchmarks > 0:
            return "FAILED"
        elif total_violations > 0:
            return "VIOLATIONS"
        elif successful_benchmarks == 0:
            return "NO_TESTS"
        else:
            return "PASSED"

    async def _save_ci_results(
        self, processed_results: dict[str, Any], suite_name: str
    ) -> None:
        """Save processed results for CI consumption."""
        # Save main results file
        results_file = self.results_dir / f"{suite_name}_results.json"
        with open(results_file, "w") as f:
            json.dump(processed_results, f, indent=2)

        # Save summary for quick CI consumption
        summary_file = self.results_dir / f"{suite_name}_summary.json"
        summary = {
            "suite_name": suite_name,
            "overall_status": processed_results["overall_status"],
            "performance_summary": processed_results["performance_summary"],
            "ci_metadata": processed_results["ci_metadata"],
        }
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        # Save violations for CI gates
        if processed_results["threshold_violations"]:
            violations_file = (
                self.results_dir / f"{suite_name}_violations.json"
            )
            with open(violations_file, "w") as f:
                json.dump(
                    processed_results["threshold_violations"], f, indent=2
                )

        self.logger.info(f"CI results saved to {self.results_dir}")

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime

        return datetime.now(UTC).isoformat().replace("+00:00", "Z")

    def validate_performance_gates(self) -> bool:
        """Validate performance gates based on results."""
        gates_enabled = self._get_env_bool("PERFORMANCE_GATES_ENABLED", True)

        if not gates_enabled:
            self.logger.info("Performance gates disabled")
            return True

        # Check for any violations in results
        violation_files = list(self.results_dir.glob("*_violations.json"))

        total_violations = 0
        for file_path in violation_files:
            try:
                with open(file_path) as f:
                    violations = json.load(f)
                    total_violations += len(violations)
            except Exception as e:
                self.logger.error(
                    f"Error reading violations from {file_path}: {e}"
                )

        if total_violations > 0:
            self.logger.error(
                f"Performance gates failed: {total_violations} violations"
            )
            return False
        else:
            self.logger.info("Performance gates passed: no violations")
            return True


async def main() -> None:
    """Main entry point for CI integration."""
    import argparse

    parser = argparse.ArgumentParser(description="Performance CI Integration")
    parser.add_argument(
        "--suite",
        default="smoke",
        choices=["smoke", "stress", "load", "endurance", "regression", "full"],
        help="Benchmark suite to run",
    )
    parser.add_argument(
        "--results-dir",
        default="performance-results",
        help="Results directory",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="benchmark-artifacts",
        help="Artifacts directory",
    )
    parser.add_argument(
        "--thresholds-config",
        default="configs/testing/performance_thresholds.yaml",
        help="Performance thresholds configuration",
    )
    parser.add_argument(
        "--validate-gates",
        action="store_true",
        help="Validate performance gates after running",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        # Initialize CI integration
        ci_integration = PerformanceCIIntegration(
            results_dir=args.results_dir,
            artifacts_dir=args.artifacts_dir,
            thresholds_config=args.thresholds_config,
        )

        # Run benchmark suite
        results = await ci_integration.run_benchmark_suite(args.suite)

        logger.info(f"Benchmark suite '{args.suite}' completed")
        logger.info(f"Overall status: {results['overall_status']}")
        logger.info(
            f"Successful benchmarks: "
            f"{results['performance_summary']['successful_benchmarks']}"
        )
        logger.info(
            f"Failed benchmarks: "
            f"{results['performance_summary']['failed_benchmarks']}"
        )
        logger.info(
            f"Total violations: "
            f"{results['performance_summary']['total_violations']}"
        )

        # Validate performance gates if requested
        if args.validate_gates:
            gates_passed = ci_integration.validate_performance_gates()
            if not gates_passed:
                logger.error("Performance gates validation failed")
                exit(1)

        logger.info("Performance CI integration completed successfully")

    except Exception as e:
        logger.error(f"Performance CI integration failed: {e}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())
