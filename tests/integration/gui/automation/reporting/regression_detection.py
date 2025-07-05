"""Regression detection engine for integration test reporting.

This module provides regression detection capabilities for identifying
performance and quality degradations in test results.
"""

from typing import Any


class RegressionDetectionEngine:
    """Engine for detecting performance and quality regressions."""

    def __init__(
        self,
        historical_data: list[dict[str, Any]],
        baseline_thresholds: dict[str, float] | None = None,
    ) -> None:
        """Initialize regression detection engine.

        Args:
            historical_data: Historical test execution data
            baseline_thresholds: Optional custom regression thresholds
        """
        self.historical_data = historical_data
        self.baseline_thresholds = (
            baseline_thresholds or self._default_thresholds()
        )

    def detect_performance_regressions(self) -> dict[str, Any]:
        """Detect performance regressions in test results.

        Returns:
            Performance regression detection results
        """
        if len(self.historical_data) < 2:
            return {"status": "insufficient_data", "regressions": []}

        regressions = {
            "response_time_regression": (
                self._detect_response_time_regression()
            ),
            "memory_usage_regression": self._detect_memory_usage_regression(),
            "throughput_regression": self._detect_throughput_regression(),
            "regression_summary": self._assess_regression_severity({}),
        }

        return regressions

    def detect_quality_regressions(self) -> dict[str, Any]:
        """Detect quality regressions in test results.

        Returns:
            Quality regression detection results
        """
        if len(self.historical_data) < 2:
            return {"status": "insufficient_data", "regressions": []}

        regressions = {
            "success_rate_regression": self._detect_success_rate_regression(),
            "error_rate_regression": self._detect_error_rate_regression(),
            "coverage_regression": self._detect_coverage_regression(),
            "stability_regression": self._detect_stability_regression(),
            "regression_summary": self._assess_regression_severity({}),
        }

        return regressions

    def generate_regression_report(self) -> dict[str, Any]:
        """Generate comprehensive regression report.

        Returns:
            Complete regression analysis report
        """
        performance_regressions = self.detect_performance_regressions()
        quality_regressions = self.detect_quality_regressions()

        return {
            "performance_regressions": performance_regressions,
            "quality_regressions": quality_regressions,
            "overall_status": self._determine_overall_regression_status(
                performance_regressions, quality_regressions
            ),
            "recommendations": self._generate_regression_recommendations(
                performance_regressions, quality_regressions
            ),
            "severity_assessment": self._assess_regression_severity(
                {**performance_regressions, **quality_regressions}
            ),
        }

    def _default_thresholds(self) -> dict[str, float]:
        """Get default regression detection thresholds."""
        return {
            "response_time_degradation_percent": 20.0,
            "memory_usage_increase_percent": 25.0,
            "success_rate_drop_percent": 5.0,
            "error_rate_increase_percent": 10.0,
            "coverage_drop_percent": 5.0,
            "stability_drop_percent": 3.0,
        }

    def _detect_response_time_regression(self) -> dict[str, Any]:
        """Detect response time regression."""
        if len(self.historical_data) < 2:
            return {"status": "insufficient_data"}

        recent_data = self.historical_data[-1]
        baseline_data = self.historical_data[-2]

        recent_response_time = recent_data.get("performance_metrics", {}).get(
            "avg_response_time_ms", 0
        )
        baseline_response_time = baseline_data.get(
            "performance_metrics", {}
        ).get("avg_response_time_ms", 0)

        if baseline_response_time == 0:
            return {"status": "no_baseline_data"}

        degradation_percent = (
            (recent_response_time - baseline_response_time)
            / baseline_response_time
        ) * 100

        threshold = self.baseline_thresholds[
            "response_time_degradation_percent"
        ]
        is_regression = degradation_percent > threshold

        return {
            "status": (
                "regression_detected" if is_regression else "no_regression"
            ),
            "recent_response_time_ms": recent_response_time,
            "baseline_response_time_ms": baseline_response_time,
            "degradation_percent": degradation_percent,
            "threshold_percent": threshold,
            "severity": (
                "high" if degradation_percent > threshold * 2 else "medium"
            ),
            "is_regression": is_regression,
        }

    def _detect_memory_usage_regression(self) -> dict[str, Any]:
        """Detect memory usage regression."""
        if len(self.historical_data) < 2:
            return {"status": "insufficient_data"}

        recent_data = self.historical_data[-1]
        baseline_data = self.historical_data[-2]

        recent_memory = recent_data.get("resource_cleanup", {}).get(
            "avg_memory_usage_mb", 0
        )
        baseline_memory = baseline_data.get("resource_cleanup", {}).get(
            "avg_memory_usage_mb", 0
        )

        if baseline_memory == 0:
            return {"status": "no_baseline_data"}

        increase_percent = (
            (recent_memory - baseline_memory) / baseline_memory
        ) * 100

        threshold = self.baseline_thresholds["memory_usage_increase_percent"]
        is_regression = increase_percent > threshold

        return {
            "status": (
                "regression_detected" if is_regression else "no_regression"
            ),
            "recent_memory_usage_mb": recent_memory,
            "baseline_memory_usage_mb": baseline_memory,
            "increase_percent": increase_percent,
            "threshold_percent": threshold,
            "severity": (
                "critical" if increase_percent > threshold * 2 else "medium"
            ),
            "is_regression": is_regression,
        }

    def _detect_throughput_regression(self) -> dict[str, Any]:
        """Detect throughput regression."""
        return {
            "status": "not_implemented",
            "message": "Throughput regression detection not yet implemented",
        }

    def _detect_success_rate_regression(self) -> dict[str, Any]:
        """Detect success rate regression."""
        if len(self.historical_data) < 2:
            return {"status": "insufficient_data"}

        recent_data = self.historical_data[-1]
        baseline_data = self.historical_data[-2]

        recent_success_rate = recent_data.get("workflow_scenarios", {}).get(
            "success_rate", 0
        )
        baseline_success_rate = baseline_data.get(
            "workflow_scenarios", {}
        ).get("success_rate", 0)

        if baseline_success_rate == 0:
            return {"status": "no_baseline_data"}

        drop_percent = baseline_success_rate - recent_success_rate

        threshold = self.baseline_thresholds["success_rate_drop_percent"]
        is_regression = drop_percent > threshold

        return {
            "status": (
                "regression_detected" if is_regression else "no_regression"
            ),
            "recent_success_rate": recent_success_rate,
            "baseline_success_rate": baseline_success_rate,
            "drop_percent": drop_percent,
            "threshold_percent": threshold,
            "severity": (
                "critical" if drop_percent > threshold * 2 else "medium"
            ),
            "is_regression": is_regression,
        }

    def _detect_error_rate_regression(self) -> dict[str, Any]:
        """Detect error rate regression."""
        return {"status": "not_implemented"}

    def _detect_coverage_regression(self) -> dict[str, Any]:
        """Detect coverage regression."""
        return {"status": "not_implemented"}

    def _detect_stability_regression(self) -> dict[str, Any]:
        """Detect stability regression."""
        return {"status": "not_implemented"}

    def _assess_regression_severity(self, regressions: dict[str, Any]) -> str:
        """Assess overall regression severity.

        Args:
            regressions: Dictionary of regression detection results

        Returns:
            Overall severity assessment
        """
        severity_levels = []

        for _regression_type, regression_data in regressions.items():
            if isinstance(regression_data, dict):
                if regression_data.get("is_regression", False):
                    severity = regression_data.get("severity", "low")
                    severity_levels.append(severity)

        if "critical" in severity_levels:
            return "critical"
        elif "high" in severity_levels:
            return "high"
        elif "medium" in severity_levels:
            return "medium"
        elif severity_levels:
            return "low"
        else:
            return "none"

    def _determine_overall_regression_status(
        self,
        performance_regressions: dict[str, Any],
        quality_regressions: dict[str, Any],
    ) -> str:
        """Determine overall regression status.

        Args:
            performance_regressions: Performance regression results
            quality_regressions: Quality regression results

        Returns:
            Overall regression status
        """
        has_performance_regression = any(
            result.get("is_regression", False)
            for result in performance_regressions.values()
            if isinstance(result, dict)
        )

        has_quality_regression = any(
            result.get("is_regression", False)
            for result in quality_regressions.values()
            if isinstance(result, dict)
        )

        if has_performance_regression and has_quality_regression:
            return "critical_regressions_detected"
        elif has_performance_regression:
            return "performance_regressions_detected"
        elif has_quality_regression:
            return "quality_regressions_detected"
        else:
            return "no_regressions_detected"

    def _generate_regression_recommendations(
        self,
        performance_regressions: dict[str, Any],
        quality_regressions: dict[str, Any],
    ) -> list[str]:
        """Generate recommendations based on detected regressions.

        Args:
            performance_regressions: Performance regression results
            quality_regressions: Quality regression results

        Returns:
            List of actionable recommendations
        """
        recommendations = []

        # Check for response time regression
        response_time_regression = performance_regressions.get(
            "response_time_regression", {}
        )
        if response_time_regression.get("is_regression", False):
            recommendations.append(
                "Investigate response time degradation - consider profiling "
                "and optimization"
            )

        # Check for memory usage regression
        memory_regression = performance_regressions.get(
            "memory_usage_regression", {}
        )
        if memory_regression.get("is_regression", False):
            recommendations.append(
                "Address memory usage increase - check for memory leaks "
                "and optimize resource usage"
            )

        # Check for success rate regression
        success_rate_regression = quality_regressions.get(
            "success_rate_regression", {}
        )
        if success_rate_regression.get("is_regression", False):
            recommendations.append(
                "Investigate test failures - review recent changes "
                "and improve test stability"
            )

        # Generic recommendations
        if not recommendations:
            recommendations.append(
                "Continue monitoring performance and quality metrics"
            )
        else:
            recommendations.append(
                "Implement additional monitoring and alerting for "
                "early regression detection"
            )

        return recommendations
