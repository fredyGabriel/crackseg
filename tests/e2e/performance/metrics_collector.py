"""Metrics collector for performance benchmark data aggregation.

This module provides utilities for collecting, aggregating, and analyzing
performance metrics from benchmark executions, supporting trend analysis
and performance regression detection.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MetricSummary:
    """Summary statistics for a performance metric."""

    name: str
    values: list[float]
    unit: str = ""

    @property
    def mean(self) -> float:
        """Mean value."""
        return statistics.mean(self.values) if self.values else 0.0

    @property
    def median(self) -> float:
        """Median value."""
        return statistics.median(self.values) if self.values else 0.0

    @property
    def std_dev(self) -> float:
        """Standard deviation."""
        return statistics.stdev(self.values) if len(self.values) > 1 else 0.0

    @property
    def min_value(self) -> float:
        """Minimum value."""
        return min(self.values) if self.values else 0.0

    @property
    def max_value(self) -> float:
        """Maximum value."""
        return max(self.values) if self.values else 0.0

    @property
    def coefficient_of_variation(self) -> float:
        """Coefficient of variation (std_dev / mean)."""
        return (self.std_dev / self.mean) if self.mean > 0 else 0.0


@dataclass
class PerformanceReport:
    """Performance analysis report."""

    benchmark_name: str
    execution_count: int
    timestamp: str
    metric_summaries: dict[str, MetricSummary]
    resource_summaries: dict[str, MetricSummary]
    threshold_violations: list[str]
    trend_analysis: dict[str, Any]


class MetricsCollector:
    """Collects and analyzes performance metrics from benchmark runs."""

    def __init__(self, storage_path: Path | None = None) -> None:
        """Initialize metrics collector."""
        self.storage_path = storage_path or Path("benchmark_metrics")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.benchmark_data: dict[str, list[dict[str, Any]]] = {}
        self.logger = logging.getLogger(__name__)

    def collect_benchmark_data(
        self,
        benchmark_name: str,
        benchmark_result: dict[str, Any],
    ) -> None:
        """Collect data from a benchmark execution."""
        if benchmark_name not in self.benchmark_data:
            self.benchmark_data[benchmark_name] = []

        # Add timestamp to the data
        data_point = {
            "timestamp": datetime.now().isoformat(),
            "benchmark_result": benchmark_result.get("benchmark_result", {}),
            "resource_metrics": benchmark_result.get("resource_metrics", {}),
        }

        self.benchmark_data[benchmark_name].append(data_point)
        self.logger.debug(f"Collected data for {benchmark_name}")

    def generate_report(self, benchmark_name: str) -> PerformanceReport:
        """Generate performance analysis report for a benchmark."""
        if benchmark_name not in self.benchmark_data:
            raise ValueError(
                f"No data available for benchmark: {benchmark_name}"
            )

        data_points = self.benchmark_data[benchmark_name]

        # Extract metrics
        metric_summaries = self._analyze_benchmark_metrics(data_points)
        resource_summaries = self._analyze_resource_metrics(data_points)

        # Collect threshold violations
        threshold_violations = []
        for data_point in data_points:
            violations = data_point["benchmark_result"].get(
                "threshold_violations", []
            )
            threshold_violations.extend(violations)

        # Perform trend analysis
        trend_analysis = self._perform_trend_analysis(data_points)

        return PerformanceReport(
            benchmark_name=benchmark_name,
            execution_count=len(data_points),
            timestamp=datetime.now().isoformat(),
            metric_summaries=metric_summaries,
            resource_summaries=resource_summaries,
            threshold_violations=list(
                set(threshold_violations)
            ),  # Remove duplicates
            trend_analysis=trend_analysis,
        )

    def _analyze_benchmark_metrics(
        self, data_points: list[dict[str, Any]]
    ) -> dict[str, MetricSummary]:
        """Analyze benchmark-specific metrics."""
        metrics_data: dict[str, list[float]] = {}

        for data_point in data_points:
            benchmark_result = data_point.get("benchmark_result", {})

            # Standard metrics
            if "success_rate" in benchmark_result:
                metrics_data.setdefault("success_rate", []).append(
                    benchmark_result["success_rate"]
                )

            if "throughput" in benchmark_result:
                metrics_data.setdefault("throughput", []).append(
                    benchmark_result["throughput"]
                )

            if "duration" in benchmark_result:
                metrics_data.setdefault("duration", []).append(
                    benchmark_result["duration"]
                )

            # Additional metrics from the metrics dict
            additional_metrics = benchmark_result.get("metrics", {})
            for metric_name, metric_value in additional_metrics.items():
                if isinstance(metric_value, int | float):
                    metrics_data.setdefault(metric_name, []).append(
                        float(metric_value)
                    )

        # Create summaries
        summaries = {}
        for metric_name, values in metrics_data.items():
            unit = self._get_metric_unit(metric_name)
            summaries[metric_name] = MetricSummary(
                name=metric_name,
                values=values,
                unit=unit,
            )

        return summaries

    def _analyze_resource_metrics(
        self, data_points: list[dict[str, Any]]
    ) -> dict[str, MetricSummary]:
        """Analyze resource usage metrics."""
        resource_data: dict[str, list[float]] = {}

        for data_point in data_points:
            resource_metrics = data_point.get("resource_metrics", {})

            # CPU metrics
            cpu_usage = resource_metrics.get("cpu_usage", {})
            if "avg" in cpu_usage:
                resource_data.setdefault("cpu_avg", []).append(
                    cpu_usage["avg"]
                )
            if "max" in cpu_usage:
                resource_data.setdefault("cpu_max", []).append(
                    cpu_usage["max"]
                )

            # Memory metrics
            memory_usage = resource_metrics.get("memory_usage_mb", {})
            if "avg" in memory_usage:
                resource_data.setdefault("memory_avg_mb", []).append(
                    memory_usage["avg"]
                )
            if "max" in memory_usage:
                resource_data.setdefault("memory_max_mb", []).append(
                    memory_usage["max"]
                )

            # GPU metrics
            gpu_memory = resource_metrics.get("gpu_memory_mb", {})
            if "avg" in gpu_memory:
                resource_data.setdefault("gpu_memory_avg_mb", []).append(
                    gpu_memory["avg"]
                )
            if "max" in gpu_memory:
                resource_data.setdefault("gpu_memory_max_mb", []).append(
                    gpu_memory["max"]
                )

        # Create summaries
        summaries = {}
        for metric_name, values in resource_data.items():
            unit = "%" if "cpu" in metric_name else "MB"
            summaries[metric_name] = MetricSummary(
                name=metric_name,
                values=values,
                unit=unit,
            )

        return summaries

    def _perform_trend_analysis(
        self, data_points: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Perform trend analysis on the data."""
        if len(data_points) < 2:
            return {"insufficient_data": True}

        # Analyze success rate trend
        success_rates = [
            dp["benchmark_result"].get("success_rate", 0) for dp in data_points
        ]

        # Analyze throughput trend
        throughputs = [
            dp["benchmark_result"].get("throughput", 0) for dp in data_points
        ]

        return {
            "success_rate_trend": self._calculate_trend(success_rates),
            "throughput_trend": self._calculate_trend(throughputs),
            "performance_stability": self._assess_stability(
                success_rates, throughputs
            ),
            "regression_detected": self._detect_regression(
                success_rates, throughputs
            ),
        }

    def _calculate_trend(self, values: list[float]) -> dict[str, Any]:
        """Calculate trend direction and strength."""
        if len(values) < 2:
            return {"direction": "unknown", "strength": 0.0}

        # Simple linear trend calculation
        n = len(values)
        x_values = list(range(n))

        # Calculate slope using least squares
        x_mean = sum(x_values) / n
        y_mean = sum(values) / n

        numerator = sum(
            (x_values[i] - x_mean) * (values[i] - y_mean) for i in range(n)
        )
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))

        slope = numerator / denominator if denominator != 0 else 0

        direction = (
            "improving"
            if slope > 0
            else "degrading" if slope < 0 else "stable"
        )
        strength = abs(slope)

        return {
            "direction": direction,
            "strength": strength,
            "slope": slope,
        }

    def _assess_stability(
        self, success_rates: list[float], throughputs: list[float]
    ) -> dict[str, Any]:
        """Assess performance stability."""
        stability = {}

        if success_rates:
            success_cv = (
                statistics.stdev(success_rates)
                / statistics.mean(success_rates)
                if statistics.mean(success_rates) > 0
                else 0
            )
            stability["success_rate_stability"] = (
                "stable" if success_cv < 0.1 else "unstable"
            )

        if throughputs:
            throughput_cv = (
                statistics.stdev(throughputs) / statistics.mean(throughputs)
                if statistics.mean(throughputs) > 0
                else 0
            )
            stability["throughput_stability"] = (
                "stable" if throughput_cv < 0.1 else "unstable"
            )

        return stability

    def _detect_regression(
        self, success_rates: list[float], throughputs: list[float]
    ) -> bool:
        """Detect performance regression."""
        if len(success_rates) < 3:
            return False

        # Check if recent performance is significantly worse
        recent_values = success_rates[-3:]
        historical_values = (
            success_rates[:-3] if len(success_rates) > 3 else []
        )

        if historical_values:
            recent_avg = statistics.mean(recent_values)
            historical_avg = statistics.mean(historical_values)

            # Regression if recent performance is 5% worse
            regression_threshold = 0.05
            return (
                historical_avg - recent_avg
            ) / historical_avg > regression_threshold

        return False

    def _get_metric_unit(self, metric_name: str) -> str:
        """Get appropriate unit for a metric."""
        units_map = {
            "success_rate": "%",
            "throughput": "ops/sec",
            "duration": "seconds",
            "avg_response_time_ms": "ms",
            "operations_per_second": "ops/sec",
            "total_runtime_seconds": "seconds",
            "operations_per_minute": "ops/min",
        }

        return units_map.get(metric_name, "")

    def export_report(
        self, report: PerformanceReport, output_path: Path
    ) -> None:
        """Export performance report to file."""
        import json

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert report to JSON-serializable format
        report_data = {
            "benchmark_name": report.benchmark_name,
            "execution_count": report.execution_count,
            "timestamp": report.timestamp,
            "metric_summaries": {
                name: {
                    "mean": summary.mean,
                    "median": summary.median,
                    "std_dev": summary.std_dev,
                    "min": summary.min_value,
                    "max": summary.max_value,
                    "unit": summary.unit,
                    "cv": summary.coefficient_of_variation,
                }
                for name, summary in report.metric_summaries.items()
            },
            "resource_summaries": {
                name: {
                    "mean": summary.mean,
                    "median": summary.median,
                    "std_dev": summary.std_dev,
                    "min": summary.min_value,
                    "max": summary.max_value,
                    "unit": summary.unit,
                }
                for name, summary in report.resource_summaries.items()
            },
            "threshold_violations": report.threshold_violations,
            "trend_analysis": report.trend_analysis,
        }

        with open(output_path, "w") as f:
            json.dump(report_data, f, indent=2)

        self.logger.info(f"Performance report exported to {output_path}")
