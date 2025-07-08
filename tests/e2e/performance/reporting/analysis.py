"""Performance analysis module for trend detection and regression analysis.

This module provides a unified interface for comprehensive performance analysis
by coordinating specialized analyzer components.
"""

from __future__ import annotations

import logging
from typing import Any

from tests.e2e.performance.reporting.insights_generator import (
    InsightsGenerator,
)
from tests.e2e.performance.reporting.regression_analyzer import (
    RegressionAnalyzer,
)
from tests.e2e.performance.reporting.trend_analyzer import TrendAnalyzer

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """Unified interface for comprehensive performance analysis."""

    def __init__(self, config: Any) -> None:
        """Initialize performance analyzer with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize specialized analyzers
        self.trend_analyzer = TrendAnalyzer()
        self.regression_analyzer = RegressionAnalyzer()
        self.insights_generator = InsightsGenerator()

    def perform_trend_analysis(
        self,
        current_data: dict[str, Any],
        historical_data: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Perform comprehensive trend analysis on performance data."""
        return self.trend_analyzer.perform_trend_analysis(
            current_data, historical_data
        )

    def analyze_metric_trend(
        self,
        metric_name: str,
        current_data: dict[str, Any],
        historical_data: list[dict[str, Any]],
        direction_preference: str,
    ) -> dict[str, Any] | None:
        """Analyze trend for a specific metric."""
        result = self.trend_analyzer.analyze_metric_trend(
            metric_name, current_data, historical_data, direction_preference
        )
        # Convert TrendData to dict if needed
        return dict(result) if result is not None else None

    def perform_regression_analysis(
        self,
        current_data: dict[str, Any],
        historical_data: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Perform regression analysis using the existing alerting system."""
        return self.regression_analyzer.perform_regression_analysis(
            current_data, historical_data
        )

    def generate_insights(
        self,
        current_data: dict[str, Any],
        trend_analysis: dict[str, Any],
        regression_analysis: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate actionable insights and recommendations."""
        return self.insights_generator.generate_insights(
            current_data, trend_analysis, regression_analysis
        )

    def perform_complete_analysis(
        self,
        current_data: dict[str, Any],
        historical_data: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Perform complete analysis including trends and regressions."""
        try:
            # Validate data
            validation = self.regression_analyzer.validate_regression_data(
                current_data, historical_data
            )

            if not validation["is_valid"]:
                self.logger.error(
                    f"Invalid data for analysis: {validation['issues']}"
                )
                return self._create_error_response(validation["issues"])

            # Log warnings if any
            for warning in validation.get("warnings", []):
                self.logger.warning(warning)

            # Perform trend analysis
            trend_analysis = self.perform_trend_analysis(
                current_data, historical_data
            )

            # Perform regression analysis
            regression_analysis = self.perform_regression_analysis(
                current_data, historical_data
            )

            # Generate insights
            insights = self.generate_insights(
                current_data, trend_analysis, regression_analysis
            )

            return {
                "trend_analysis": trend_analysis,
                "regression_analysis": regression_analysis,
                "insights_and_recommendations": insights,
                "analysis_metadata": {
                    "data_points_analyzed": len(historical_data),
                    "validation_warnings": validation.get("warnings", []),
                },
            }

        except Exception as e:
            self.logger.error(f"Complete analysis failed: {e}")
            return self._create_error_response([str(e)])

    def _create_error_response(self, errors: list[str]) -> dict[str, Any]:
        """Create error response when analysis fails."""
        return {
            "trend_analysis": {
                "trends": [],
                "summary": "Analysis failed",
                "data_points": 0,
                "confidence": 0.0,
            },
            "regression_analysis": {
                "regressions_detected": 0,
                "severity_breakdown": {
                    "low": 0,
                    "medium": 0,
                    "high": 0,
                    "critical": 0,
                },
                "detailed_regressions": [],
            },
            "insights_and_recommendations": {
                "summary": "Analysis failed due to errors",
                "key_findings": errors,
                "recommendations": ["Fix data issues and retry analysis"],
                "risk_assessment": "unknown",
            },
            "analysis_metadata": {
                "data_points_analyzed": 0,
                "errors": errors,
            },
        }

    def get_analysis_summary(self, complete_analysis: dict[str, Any]) -> str:
        """Get a brief summary of the complete analysis."""
        trend_summary = complete_analysis.get("trend_analysis", {}).get(
            "summary", ""
        )
        regression_summary = self.regression_analyzer.get_regression_summary(
            complete_analysis.get("regression_analysis", {})
        )
        risk_level = complete_analysis.get(
            "insights_and_recommendations", {}
        ).get("risk_assessment", "unknown")

        return (
            f"Risk: {risk_level.title()} | {trend_summary} | "
            f"{regression_summary}"
        )
