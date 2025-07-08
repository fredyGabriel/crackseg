"""Data processing module for performance benchmark results.

This module handles the conversion of raw benchmark results into standardized
performance data structures for analysis and reporting.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


class BenchmarkDataProcessor:
    """Processes raw benchmark results into standardized performance data."""

    def __init__(self) -> None:
        """Initialize the benchmark data processor."""
        self.logger = logging.getLogger(__name__)

    def process_benchmark_results(
        self, results: dict[str, Any]
    ) -> dict[str, Any]:
        """Process raw benchmark results into standardized performance data."""
        processed: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "overall_summary": self._create_empty_summary(),
            "benchmark_details": {},
            "resource_summary": self._create_empty_resource_summary(),
        }

        # Extract benchmark data with error handling
        benchmark_results = results.get("benchmark_results", {})
        if not benchmark_results:
            self.logger.warning("No benchmark results found in input data")
            return processed

        # Process individual benchmarks
        metrics = self._extract_all_metrics(benchmark_results)

        # Calculate summaries
        processed["overall_summary"] = self._calculate_overall_summary(
            benchmark_results, metrics
        )
        processed["resource_summary"] = self._calculate_resource_summary(
            metrics
        )
        processed["benchmark_details"] = self._create_benchmark_details(
            benchmark_results
        )

        return processed

    def _create_empty_summary(self) -> dict[str, Any]:
        """Create empty overall summary structure."""
        return {
            "total_benchmarks": 0,
            "successful_benchmarks": 0,
            "average_success_rate": 0.0,
            "average_throughput": 0.0,
            "total_violations": 0,
        }

    def _create_empty_resource_summary(self) -> dict[str, Any]:
        """Create empty resource summary structure."""
        return {
            "peak_memory_mb": 0.0,
            "avg_cpu_usage": 0.0,
            "max_response_time": 0.0,
        }

    def _extract_all_metrics(
        self, benchmark_results: dict[str, Any]
    ) -> dict[str, list[float]]:
        """Extract all metrics from benchmark results."""
        metrics = {
            "success_rates": [],
            "throughputs": [],
            "memory_usage": [],
            "cpu_usage": [],
            "response_times": [],
            "violations": [],
        }

        for result in benchmark_results.values():
            # Extract basic metrics
            success_rate = result.get("success_rate", 0.0)
            throughput = result.get("throughput", 0.0)
            violations = result.get("threshold_violations", [])

            metrics["success_rates"].append(success_rate)
            metrics["throughputs"].append(throughput)
            metrics["violations"].append(len(violations))

            # Extract resource metrics
            result_metrics = result.get("metrics", {})
            response_time = result_metrics.get("average_response_time", 0.0)
            peak_memory = result_metrics.get("peak_memory_mb", 0.0)

            metrics["response_times"].append(response_time)
            metrics["memory_usage"].append(peak_memory)

            # Extract CPU usage
            resource_metrics = result.get("resource_metrics", {})
            cpu_info = resource_metrics.get("cpu_usage", {})
            cpu_avg = cpu_info.get("avg", 0.0)
            metrics["cpu_usage"].append(cpu_avg)

        return metrics

    def _calculate_overall_summary(
        self,
        benchmark_results: dict[str, Any],
        metrics: dict[str, list[float]],
    ) -> dict[str, Any]:
        """Calculate overall performance summary."""
        success_rates = metrics["success_rates"]
        throughputs = metrics["throughputs"]
        violations = metrics["violations"]

        if not success_rates:
            return self._create_empty_summary()

        return {
            "total_benchmarks": len(benchmark_results),
            "successful_benchmarks": sum(
                1 for sr in success_rates if sr >= 95
            ),
            "average_success_rate": sum(success_rates) / len(success_rates),
            "average_throughput": (
                sum(throughputs) / len(throughputs) if throughputs else 0.0
            ),
            "total_violations": sum(violations),
        }

    def _calculate_resource_summary(
        self, metrics: dict[str, list[float]]
    ) -> dict[str, Any]:
        """Calculate resource usage summary."""
        memory_usage = metrics["memory_usage"]
        cpu_usage = metrics["cpu_usage"]
        response_times = metrics["response_times"]

        return {
            "peak_memory_mb": max(memory_usage) if memory_usage else 0.0,
            "avg_cpu_usage": (
                sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0.0
            ),
            "max_response_time": (
                max(response_times) if response_times else 0.0
            ),
        }

    def _create_benchmark_details(
        self, benchmark_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Create detailed breakdown of individual benchmark results."""
        details = {}

        for benchmark_name, result in benchmark_results.items():
            # Extract basic metrics
            success_rate = result.get("success_rate", 0.0)
            throughput = result.get("throughput", 0.0)
            violations = result.get("threshold_violations", [])

            # Extract detailed metrics
            result_metrics = result.get("metrics", {})
            response_time = result_metrics.get("average_response_time", 0.0)
            peak_memory = result_metrics.get("peak_memory_mb", 0.0)

            # Extract CPU usage
            resource_metrics = result.get("resource_metrics", {})
            cpu_info = resource_metrics.get("cpu_usage", {})
            cpu_avg = cpu_info.get("avg", 0.0)

            details[benchmark_name] = {
                "success_rate": success_rate,
                "throughput": throughput,
                "response_time": response_time,
                "peak_memory_mb": peak_memory,
                "cpu_avg": cpu_avg,
                "violations": violations,
                "violation_count": len(violations),
            }

        return details

    def validate_benchmark_results(
        self, results: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate benchmark results structure and content."""
        validation: dict[str, Any] = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
        }

        # Check basic structure
        if not isinstance(results, dict):
            validation["is_valid"] = False
            validation["errors"].append("Results must be a dictionary")
            return validation

        # Check for benchmark_results key
        if "benchmark_results" not in results:
            validation["warnings"].append("No 'benchmark_results' key found")
            return validation

        benchmark_results = results["benchmark_results"]
        if not isinstance(benchmark_results, dict):
            validation["is_valid"] = False
            validation["errors"].append(
                "benchmark_results must be a dictionary"
            )
            return validation

        # Validate individual benchmarks
        for benchmark_name, result in benchmark_results.items():
            benchmark_validation = self._validate_single_benchmark(
                benchmark_name, result
            )
            validation["warnings"].extend(benchmark_validation["warnings"])
            validation["errors"].extend(benchmark_validation["errors"])
            if not benchmark_validation["is_valid"]:
                validation["is_valid"] = False

        return validation

    def _validate_single_benchmark(
        self, name: str, result: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate a single benchmark result."""
        validation: dict[str, Any] = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
        }

        required_fields = ["success_rate", "throughput"]
        for field in required_fields:
            if field not in result:
                validation["warnings"].append(
                    f"Benchmark '{name}' missing field: {field}"
                )

        # Check data types
        if "success_rate" in result:
            success_rate = result["success_rate"]
            if (
                not isinstance(success_rate, int | float)
                or success_rate < 0
                or success_rate > 100
            ):
                validation["errors"].append(
                    f"Benchmark '{name}' has invalid success_rate: "
                    f"{success_rate}"
                )
                validation["is_valid"] = False

        return validation
