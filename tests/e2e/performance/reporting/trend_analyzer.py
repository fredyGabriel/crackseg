"""Trend analysis module for performance metrics.

This module provides comprehensive trend analysis capabilities including
trend detection, statistical confidence calculations, and metric evolution
tracking.
"""

from __future__ import annotations

import logging
import statistics
from typing import Any

from tests.e2e.performance.reporting.config import (
    TREND_MIN_DATA_POINTS,
    TREND_STABLE_THRESHOLD,
    TrendData,
)
from tests.e2e.performance.reporting.metric_extractor import MetricExtractor

logger = logging.getLogger(__name__)


class TrendAnalyzer:
    """Handles trend analysis for performance metrics."""

    def __init__(self) -> None:
        """Initialize trend analyzer."""
        self.metric_extractor = MetricExtractor()
        self.logger = logging.getLogger(__name__)

    def perform_trend_analysis(
        self,
        current_data: dict[str, Any],
        historical_data: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Perform comprehensive trend analysis on performance data."""
        if len(historical_data) < TREND_MIN_DATA_POINTS:
            self.logger.warning(
                f"Insufficient historical data for trend analysis: "
                f"{len(historical_data)} points"
            )
            return {
                "trends": [],
                "summary": "Insufficient data for trend analysis",
                "data_points": len(historical_data),
                "confidence": 0.0,
            }

        # Key metrics to analyze
        metrics_to_analyze = [
            ("average_success_rate", "higher"),
            ("average_throughput", "higher"),
            ("total_violations", "lower"),
            ("peak_memory_mb", "lower"),
            ("avg_cpu_usage", "lower"),
        ]

        trends = []
        for metric_name, direction_preference in metrics_to_analyze:
            trend = self.analyze_metric_trend(
                metric_name,
                current_data,
                historical_data,
                direction_preference,
            )
            if trend:
                trends.append(trend)

        # Calculate overall trend confidence
        overall_confidence = (
            statistics.mean([t["confidence"] for t in trends])
            if trends
            else 0.0
        )

        return {
            "trends": trends,
            "summary": self._generate_trend_summary(trends),
            "data_points": len(historical_data),
            "confidence": overall_confidence,
        }

    def analyze_metric_trend(
        self,
        metric_name: str,
        current_data: dict[str, Any],
        historical_data: list[dict[str, Any]],
        direction_preference: str,
    ) -> TrendData | None:
        """Analyze trend for a specific metric."""
        current_value = self.metric_extractor.extract_metric_value(
            current_data, metric_name
        )
        if current_value is None:
            return None

        # Extract historical values
        historical_values = []
        for data_point in historical_data:
            value = self.metric_extractor.extract_metric_value(
                data_point, metric_name
            )
            if value is not None:
                historical_values.append(value)

        if len(historical_values) < TREND_MIN_DATA_POINTS:
            return None

        # Calculate trend
        recent_avg = statistics.mean(historical_values[-3:])
        percentage_change = (
            ((current_value - recent_avg) / recent_avg * 100)
            if recent_avg != 0
            else 0.0
        )

        # Determine trend direction
        trend_direction = self._determine_trend_direction(
            percentage_change, direction_preference
        )

        # Calculate confidence
        confidence = self._calculate_trend_confidence(
            historical_values, current_value
        )

        # Generate recommendation
        recommendation = self._generate_trend_recommendation(
            metric_name, trend_direction, percentage_change, confidence
        )

        return TrendData(
            metric_name=metric_name,
            trend_direction=trend_direction,
            trend_percentage=percentage_change,
            confidence=confidence,
            recommendation=recommendation,
        )

    def _determine_trend_direction(
        self, percentage_change: float, direction_preference: str
    ) -> str:
        """Determine trend direction based on percentage change."""
        if abs(percentage_change) < TREND_STABLE_THRESHOLD:
            return "stable"
        elif percentage_change > 0:
            return (
                "improving"
                if direction_preference == "higher"
                else "degrading"
            )
        else:
            return (
                "degrading"
                if direction_preference == "higher"
                else "improving"
            )

    def _calculate_trend_confidence(
        self, historical_values: list[float], current_value: float
    ) -> float:
        """Calculate confidence score for trend analysis."""
        if len(historical_values) < 2:
            return 0.0

        # Calculate variability in historical data
        try:
            std_dev = statistics.stdev(historical_values)
            mean_val = statistics.mean(historical_values)

            # Coefficient of variation
            cv = std_dev / mean_val if mean_val != 0 else 0.0

            # Lower variability = higher confidence
            base_confidence = max(0.0, 1.0 - cv)

            # Adjust for data points (more points = higher confidence)
            data_points_factor = min(1.0, len(historical_values) / 10)

            # Adjust for how far current value is from historical mean
            deviation_factor = 1.0
            if std_dev > 0:
                z_score = abs(current_value - mean_val) / std_dev
                deviation_factor = max(0.1, 1.0 - (z_score / 3.0))

            confidence = (
                base_confidence * data_points_factor * deviation_factor
            )
            return max(0.0, min(1.0, confidence))

        except (statistics.StatisticsError, ZeroDivisionError):
            return 0.0

    def _generate_trend_recommendation(
        self,
        metric_name: str,
        trend_direction: str,
        percentage_change: float,
        confidence: float,
    ) -> str:
        """Generate recommendation based on trend analysis."""
        if confidence < 0.3:
            return (
                f"Trend for {metric_name} is uncertain. "
                f"Collect more data points for reliable analysis."
            )

        if trend_direction == "stable":
            return (
                f"{metric_name} is stable. "
                f"Continue current performance practices."
            )

        if trend_direction == "improving":
            return (
                f"{metric_name} is improving by "
                f"{abs(percentage_change):.1f}%. "
                f"Identify and maintain successful practices."
            )

        # Degrading trend
        severity = self._determine_severity(percentage_change)
        return (
            f"{metric_name} is degrading by {abs(percentage_change):.1f}% "
            f"({severity}). Investigate root cause and implement corrective "
            f"measures."
        )

    def _determine_severity(self, percentage_change: float) -> str:
        """Determine severity level based on percentage change."""
        abs_change = abs(percentage_change)
        if abs_change > 20:
            return "critical"
        elif abs_change > 10:
            return "moderate"
        else:
            return "minor"

    def _generate_trend_summary(self, trends: list[TrendData]) -> str:
        """Generate summary of trend analysis results."""
        if not trends:
            return "No trend data available"

        improving = sum(
            1 for t in trends if t["trend_direction"] == "improving"
        )
        degrading = sum(
            1 for t in trends if t["trend_direction"] == "degrading"
        )
        stable = sum(1 for t in trends if t["trend_direction"] == "stable")

        return (
            f"Performance trends: {improving} improving, {stable} stable, "
            f"{degrading} degrading"
        )
