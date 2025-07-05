"""Specific trend analyzers for different metrics.

This module contains focused trend analyzer methods that analyze specific
types of metrics like response time, memory usage, success rates, etc.
"""

from typing import Any


class TrendAnalyzers:
    """Collection of specific trend analyzer methods."""

    def __init__(self, historical_data: list[dict[str, Any]]) -> None:
        """Initialize trend analyzers.

        Args:
            historical_data: Historical test execution data
        """
        self.historical_data = historical_data

    def analyze_response_time_trend(self) -> dict[str, Any]:
        """Analyze response time trends."""
        response_times = []
        timestamps = []

        for data_point in self.historical_data:
            performance_data = data_point.get("performance_metrics", {})
            response_time = performance_data.get("avg_response_time_ms", 0)
            response_times.append(response_time)
            timestamps.append(data_point.get("timestamp", ""))

        if len(response_times) < 2:
            return {"trend": "insufficient_data"}

        # Calculate trend direction
        recent_avg = sum(response_times[-3:]) / min(3, len(response_times))
        historical_avg = sum(response_times[:-3]) / max(
            1, len(response_times) - 3
        )

        trend_direction = "stable"
        if recent_avg > historical_avg * 1.1:
            trend_direction = "increasing"
        elif recent_avg < historical_avg * 0.9:
            trend_direction = "decreasing"

        return {
            "trend": trend_direction,
            "current_avg": recent_avg,
            "historical_avg": historical_avg,
            "data_points": len(response_times),
            "min_response_time": min(response_times),
            "max_response_time": max(response_times),
        }

    def analyze_memory_usage_trend(self) -> dict[str, Any]:
        """Analyze memory usage trends."""
        memory_usage = []

        for data_point in self.historical_data:
            resource_data = data_point.get("resource_cleanup", {})
            memory = resource_data.get("avg_memory_usage_mb", 0)
            memory_usage.append(memory)

        if len(memory_usage) < 2:
            return {"trend": "insufficient_data"}

        recent_avg = sum(memory_usage[-3:]) / min(3, len(memory_usage))
        historical_avg = sum(memory_usage[:-3]) / max(1, len(memory_usage) - 3)

        trend_direction = "stable"
        if recent_avg > historical_avg * 1.1:
            trend_direction = "increasing"
        elif recent_avg < historical_avg * 0.9:
            trend_direction = "decreasing"

        return {
            "trend": trend_direction,
            "current_avg_mb": recent_avg,
            "historical_avg_mb": historical_avg,
            "peak_usage_mb": max(memory_usage),
            "efficiency_score": self._calculate_memory_efficiency_score(
                memory_usage
            ),
        }

    def analyze_success_rate_trend(self) -> dict[str, Any]:
        """Analyze success rate trends."""
        success_rates = []

        for data_point in self.historical_data:
            workflow_data = data_point.get("workflow_scenarios", {})
            success_rate = workflow_data.get("success_rate", 0)
            success_rates.append(success_rate)

        if len(success_rates) < 2:
            return {"trend": "insufficient_data"}

        recent_avg = sum(success_rates[-3:]) / min(3, len(success_rates))
        historical_avg = sum(success_rates[:-3]) / max(
            1, len(success_rates) - 3
        )

        trend_direction = "stable"
        if recent_avg > historical_avg + 2.0:
            trend_direction = "improving"
        elif recent_avg < historical_avg - 2.0:
            trend_direction = "declining"

        return {
            "trend": trend_direction,
            "current_success_rate": recent_avg,
            "historical_success_rate": historical_avg,
            "best_success_rate": max(success_rates),
            "worst_success_rate": min(success_rates),
            "consistency_score": self._calculate_consistency_score(
                success_rates
            ),
        }

    def analyze_resource_efficiency_trend(self) -> dict[str, Any]:
        """Analyze resource efficiency trends."""
        efficiency_scores = []

        for data_point in self.historical_data:
            resource_data = data_point.get("resource_cleanup", {})
            cleanup_rate = resource_data.get("cleanup_success_rate", 0)
            efficiency_scores.append(cleanup_rate)

        if len(efficiency_scores) < 2:
            return {"trend": "insufficient_data"}

        recent_avg = sum(efficiency_scores[-3:]) / min(
            3, len(efficiency_scores)
        )
        historical_avg = sum(efficiency_scores[:-3]) / max(
            1, len(efficiency_scores) - 3
        )

        trend_direction = "stable"
        if recent_avg > historical_avg + 1.0:
            trend_direction = "improving"
        elif recent_avg < historical_avg - 1.0:
            trend_direction = "declining"

        return {
            "trend": trend_direction,
            "current_efficiency": recent_avg,
            "historical_efficiency": historical_avg,
            "peak_efficiency": max(efficiency_scores),
            "efficiency_volatility": self._calculate_volatility(
                efficiency_scores
            ),
        }

    def analyze_test_coverage_trend(self) -> dict[str, Any]:
        """Analyze test coverage trends."""
        coverage_values = []

        for data_point in self.historical_data:
            workflow_data = data_point.get("workflow_scenarios", {})
            coverage = workflow_data.get("coverage_percentage", 0)
            coverage_values.append(coverage)

        if len(coverage_values) < 2:
            return {"trend": "insufficient_data"}

        recent_avg = sum(coverage_values[-3:]) / min(3, len(coverage_values))
        historical_avg = sum(coverage_values[:-3]) / max(
            1, len(coverage_values) - 3
        )

        return {
            "trend": "improving" if recent_avg > historical_avg else "stable",
            "current_coverage": recent_avg,
            "target_coverage": 95.0,
            "coverage_gap": max(0, 95.0 - recent_avg),
        }

    def analyze_error_rate_trend(self) -> dict[str, Any]:
        """Analyze error rate trends."""
        error_rates = []

        for data_point in self.historical_data:
            error_data = data_point.get("error_scenarios", {})
            handled = error_data.get("handled_gracefully", 0)
            total = error_data.get("total_error_scenarios", 1)
            error_rate = (total - handled) / total * 100
            error_rates.append(error_rate)

        if len(error_rates) < 2:
            return {"trend": "insufficient_data"}

        recent_avg = sum(error_rates[-3:]) / min(3, len(error_rates))
        historical_avg = sum(error_rates[:-3]) / max(1, len(error_rates) - 3)

        trend_direction = "stable"
        if recent_avg < historical_avg - 1.0:
            trend_direction = "improving"  # Lower error rate is better
        elif recent_avg > historical_avg + 1.0:
            trend_direction = "worsening"

        return {
            "trend": trend_direction,
            "current_error_rate": recent_avg,
            "historical_error_rate": historical_avg,
            "target_error_rate": 5.0,
        }

    def analyze_automation_reliability_trend(self) -> dict[str, Any]:
        """Analyze automation reliability trends."""
        reliability_scores = []

        for data_point in self.historical_data:
            automation_data = data_point.get("automation_metrics", {})
            success_rate = automation_data.get("automation_success_rate", 0)
            reliability_scores.append(success_rate)

        if len(reliability_scores) < 2:
            return {"trend": "insufficient_data"}

        recent_avg = sum(reliability_scores[-3:]) / min(
            3, len(reliability_scores)
        )

        return {
            "trend": "stable",  # Automation should be consistently high
            "current_reliability": recent_avg,
            "target_reliability": 99.0,
            "reliability_variance": self._calculate_variance(
                reliability_scores
            ),
        }

    def analyze_stability_trend(self) -> dict[str, Any]:
        """Analyze system stability trends."""
        stability_scores = []

        for data_point in self.historical_data:
            concurrent_data = data_point.get("concurrent_operations", {})
            stability = concurrent_data.get("stability_rate", 0)
            stability_scores.append(stability)

        if len(stability_scores) < 2:
            return {"trend": "insufficient_data"}

        recent_avg = sum(stability_scores[-3:]) / min(3, len(stability_scores))

        return {
            "trend": "stable" if recent_avg > 95.0 else "needs_attention",
            "current_stability": recent_avg,
            "target_stability": 99.0,
            "stability_consistency": self._calculate_consistency_score(
                stability_scores
            ),
        }

    # Utility methods for statistical calculations
    def _calculate_memory_efficiency_score(
        self, memory_usage: list[float]
    ) -> float:
        """Calculate memory efficiency score."""
        if not memory_usage:
            return 0.0

        avg_usage = sum(memory_usage) / len(memory_usage)
        # Lower memory usage gets higher efficiency score
        return max(0.0, 100.0 - (avg_usage / 1000.0) * 10)

    def _calculate_consistency_score(self, values: list[float]) -> float:
        """Calculate consistency score (inverse of variance)."""
        if len(values) < 2:
            return 100.0

        mean_val = sum(values) / len(values)
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        # Convert to 0-100 scale where 100 is most consistent
        return max(0.0, 100.0 - variance)

    def _calculate_volatility(self, values: list[float]) -> float:
        """Calculate volatility of values."""
        if len(values) < 2:
            return 0.0

        return self._calculate_variance(values) ** 0.5

    def _calculate_variance(self, values: list[float]) -> float:
        """Calculate variance of values."""
        if len(values) < 2:
            return 0.0

        mean_val = sum(values) / len(values)
        return sum((x - mean_val) ** 2 for x in values) / len(values)
