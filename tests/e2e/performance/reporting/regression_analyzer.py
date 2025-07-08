"""Regression analysis module for performance monitoring.

This module provides regression analysis capabilities by integrating with
the existing regression alerting system.
"""

from __future__ import annotations

import logging
from typing import Any

from tests.e2e.performance.regression_alerting_system import (
    RegressionAnalyzer as SystemRegressionAnalyzer,
)
from tests.e2e.performance.regression_alerting_system import (
    RegressionMetric,
    RegressionThresholds,
)

logger = logging.getLogger(__name__)


class RegressionAnalyzer:
    """Handles performance regression analysis using the alerting system."""

    def __init__(self) -> None:
        """Initialize regression analyzer."""
        self.system_analyzer = SystemRegressionAnalyzer(RegressionThresholds())
        self.logger = logging.getLogger(__name__)

    def perform_regression_analysis(
        self,
        current_data: dict[str, Any],
        historical_data: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Perform regression analysis using the existing alerting system."""
        try:
            regressions = self.system_analyzer.analyze_performance_data(
                current_data, historical_data
            )
        except Exception as e:
            self.logger.error(f"Regression analysis failed: {e}")
            return self._create_error_analysis(str(e))

        analysis = self._process_regressions(regressions)
        return analysis

    def _process_regressions(
        self, regressions: list[RegressionMetric]
    ) -> dict[str, Any]:
        """Process regression results into structured analysis."""
        analysis: dict[str, Any] = {
            "regressions_detected": len(regressions),
            "severity_breakdown": {
                "low": 0,
                "medium": 0,
                "high": 0,
                "critical": 0,
            },
            "detailed_regressions": [],
            "detected_regressions": [],
            "performance_improvements": [],
        }

        severity_breakdown: dict[str, int] = analysis["severity_breakdown"]
        detailed_regressions: list[Any] = analysis["detailed_regressions"]
        detected_regressions: list[Any] = analysis["detected_regressions"]
        performance_improvements: list[Any] = analysis[
            "performance_improvements"
        ]

        for regression in regressions:
            severity = regression["severity"]
            if severity in severity_breakdown:
                severity_breakdown[severity] += 1

            # Convert to dict for detailed regressions
            detailed_regression = dict(regression)
            detailed_regressions.append(detailed_regression)

            # Classify as regression or improvement
            impact = regression["change_percentage"]
            if impact < 0:  # Negative impact is actually an improvement
                performance_improvements.append(
                    {
                        "metric": regression["metric_name"],
                        "impact_percentage": abs(impact),
                        "description": (
                            f"Improved {regression['metric_name']} by "
                            f"{abs(impact):.1f}%"
                        ),
                    }
                )
            else:
                detected_regressions.append(
                    {
                        "metric": regression["metric_name"],
                        "impact_percentage": impact,
                        "severity": severity,
                        "description": (
                            f"Degraded {regression['metric_name']} by "
                            f"{impact:.1f}%"
                        ),
                    }
                )

        return analysis

    def _create_error_analysis(self, error_message: str) -> dict[str, Any]:
        """Create error analysis when regression detection fails."""
        return {
            "regressions_detected": 0,
            "severity_breakdown": {
                "low": 0,
                "medium": 0,
                "high": 0,
                "critical": 0,
            },
            "detailed_regressions": [],
            "detected_regressions": [],
            "performance_improvements": [],
            "error": error_message,
        }

    def get_regression_summary(self, analysis: dict[str, Any]) -> str:
        """Generate summary of regression analysis."""
        regressions_count = analysis.get("regressions_detected", 0)
        severity_breakdown = analysis.get("severity_breakdown", {})

        if regressions_count == 0:
            return "No performance regressions detected"

        critical = severity_breakdown.get("critical", 0)
        high = severity_breakdown.get("high", 0)
        medium = severity_breakdown.get("medium", 0)
        low = severity_breakdown.get("low", 0)

        parts = []
        if critical > 0:
            parts.append(f"{critical} critical")
        if high > 0:
            parts.append(f"{high} high")
        if medium > 0:
            parts.append(f"{medium} medium")
        if low > 0:
            parts.append(f"{low} low")

        severity_str = ", ".join(parts)
        return f"{regressions_count} regressions detected: {severity_str}"

    def validate_regression_data(
        self,
        current_data: dict[str, Any],
        historical_data: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Validate data for regression analysis."""
        validation: dict[str, Any] = {
            "is_valid": True,
            "issues": [],
            "warnings": [],
        }

        # Check current data structure
        if not isinstance(current_data, dict):
            validation["is_valid"] = False
            validation["issues"].append("Current data is not a dictionary")

        # Check historical data
        if not isinstance(historical_data, list):
            validation["is_valid"] = False
            validation["issues"].append("Historical data is not a list")
        elif len(historical_data) < 2:
            validation["warnings"].append(
                f"Limited historical data: {len(historical_data)} points"
            )

        # Check for required fields
        required_fields = ["overall_summary", "resource_summary"]
        for field in required_fields:
            if field not in current_data:
                validation["warnings"].append(
                    f"Missing field in current data: {field}"
                )

        return validation
