"""Trend analysis engine for integration test reporting.

This module provides trend analysis capabilities for historical test data,
including performance trends, quality trends, and future predictions.
"""

from typing import Any

from .trend_analyzers import TrendAnalyzers
from .trend_predictions import TrendPredictor


class TrendAnalysisEngine:
    """Engine for performing trend analysis on historical test data."""

    def __init__(self, historical_data: list[dict[str, Any]]) -> None:
        """Initialize trend analysis engine.

        Args:
            historical_data: Historical test execution data
        """
        self.historical_data = historical_data
        self.analyzers = TrendAnalyzers(historical_data)
        self.predictor = TrendPredictor(historical_data)

    def analyze_performance_trends(self) -> dict[str, Any]:
        """Analyze performance trends across historical data.

        Returns:
            Performance trend analysis results
        """
        if len(self.historical_data) < 2:
            return {
                "status": "insufficient_data",
                "data_points": len(self.historical_data),
            }

        performance_trends = {
            "response_time_trend": (
                self.analyzers.analyze_response_time_trend()
            ),
            "memory_usage_trend": self.analyzers.analyze_memory_usage_trend(),
            "success_rate_trend": self.analyzers.analyze_success_rate_trend(),
            "resource_efficiency_trend": (
                self.analyzers.analyze_resource_efficiency_trend()
            ),
            "trend_summary": self.predictor.generate_trend_summary(),
        }

        return performance_trends

    def analyze_quality_trends(self) -> dict[str, Any]:
        """Analyze quality trends across test executions.

        Returns:
            Quality trend analysis results
        """
        if len(self.historical_data) < 2:
            return {
                "status": "insufficient_data",
                "data_points": len(self.historical_data),
            }

        quality_trends = {
            "test_coverage_trend": (
                self.analyzers.analyze_test_coverage_trend()
            ),
            "error_rate_trend": self.analyzers.analyze_error_rate_trend(),
            "automation_reliability_trend": (
                self.analyzers.analyze_automation_reliability_trend()
            ),
            "stability_trend": self.analyzers.analyze_stability_trend(),
            "quality_summary": self.predictor.generate_quality_summary(),
        }

        return quality_trends

    def predict_future_trends(
        self, prediction_horizon_days: int = 30
    ) -> dict[str, Any]:
        """Predict future trends based on historical patterns.

        Args:
            prediction_horizon_days: Number of days to predict ahead

        Returns:
            Future trend predictions
        """
        return self.predictor.predict_future_trends(prediction_horizon_days)
