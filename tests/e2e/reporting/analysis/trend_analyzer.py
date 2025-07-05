"""Trend analysis module for test execution patterns.

This module provides trend analysis, regression detection, and predictive
insights based on historical test execution data and performance metrics.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, TypedDict

logger = logging.getLogger(__name__)


class TrendData(TypedDict):
    """Type definition for trend analysis data."""

    metric_name: str
    historical_values: list[tuple[str, float]]  # (timestamp, value)
    current_value: float
    trend_direction: str  # "improving", "degrading", "stable"
    change_percentage: float
    significance: str  # "critical", "warning", "info"


class TestInsight(TypedDict):
    """Type definition for actionable test insights."""

    insight_type: str
    title: str
    description: str
    severity: str
    recommended_actions: list[str]
    affected_tests: list[str]


class TestTrendAnalyzer:
    """Analyzer for test execution trends and performance patterns.

    Provides trend analysis, regression detection, and predictive insights
    based on historical test execution data.
    """

    def __init__(self, historical_data_path: Path | None = None) -> None:
        """Initialize the trend analyzer.

        Args:
            historical_data_path: Path to historical test data
        """
        self.historical_data_path = historical_data_path or Path(
            "test-reports/historical"
        )
        self.historical_data_path.mkdir(parents=True, exist_ok=True)

        logger.debug("Trend analyzer initialized")

    def analyze_trends(
        self, current_summary: dict[str, Any], lookback_days: int = 30
    ) -> dict[str, Any]:
        """Analyze trends in test execution metrics.

        Args:
            current_summary: Current execution summary
            lookback_days: Number of days to look back for trend analysis

        Returns:
            Dictionary containing trend analysis results
        """
        historical_data = self._load_historical_data(lookback_days)

        if len(historical_data) < 2:
            return {
                "trends": [],
                "insights": [],
                "regression_alerts": [],
                "message": "Insufficient historical data for trend analysis",
            }

        trends = []

        # Analyze success rate trend
        success_rate_trend = self._analyze_metric_trend(
            historical_data,
            "success_rate",
            current_summary.get("success_rate", 0),
        )
        trends.append(success_rate_trend)

        # Analyze execution time trend
        duration_trend = self._analyze_metric_trend(
            historical_data,
            "total_duration",
            current_summary.get("total_duration", 0),
        )
        trends.append(duration_trend)

        # Generate insights
        insights = self._generate_trend_insights(trends)

        # Check for regressions
        regression_alerts = self._detect_regressions(trends)

        return {
            "trends": trends,
            "insights": insights,
            "regression_alerts": regression_alerts,
            "historical_data_points": len(historical_data),
        }

    def _load_historical_data(
        self, lookback_days: int
    ) -> list[dict[str, Any]]:
        """Load historical execution data."""
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        historical_data = []

        if not self.historical_data_path.exists():
            return historical_data

        for data_file in self.historical_data_path.glob(
            "execution_summary_*.json"
        ):
            try:
                # Extract timestamp from filename
                timestamp_str = data_file.stem.split("_")[-2:]
                if len(timestamp_str) == 2:
                    timestamp = datetime.strptime(
                        "_".join(timestamp_str), "%Y%m%d_%H%M%S"
                    )
                    if timestamp >= cutoff_date:
                        with open(data_file, encoding="utf-8") as f:
                            data = json.load(f)
                            data["timestamp"] = timestamp.isoformat()
                            historical_data.append(data)
            except (ValueError, json.JSONDecodeError, KeyError):
                logger.debug(
                    f"Skipping invalid historical data file: {data_file}"
                )

        return sorted(historical_data, key=lambda x: x["timestamp"])

    def _analyze_metric_trend(
        self,
        historical_data: list[dict[str, Any]],
        metric_name: str,
        current_value: float,
    ) -> TrendData:
        """Analyze trend for a specific metric."""
        historical_values = []

        for data_point in historical_data:
            if metric_name in data_point:
                historical_values.append(
                    (data_point["timestamp"], data_point[metric_name])
                )

        if len(historical_values) < 2:
            return TrendData(
                metric_name=metric_name,
                historical_values=historical_values,
                current_value=current_value,
                trend_direction="stable",
                change_percentage=0.0,
                significance="info",
            )

        # Calculate trend direction
        recent_avg = sum(v[1] for v in historical_values[-3:]) / min(
            3, len(historical_values)
        )
        older_avg = sum(v[1] for v in historical_values[:3]) / min(
            3, len(historical_values)
        )

        if recent_avg > older_avg * 1.05:
            trend_direction = (
                "improving" if metric_name == "success_rate" else "degrading"
            )
        elif recent_avg < older_avg * 0.95:
            trend_direction = (
                "degrading" if metric_name == "success_rate" else "improving"
            )
        else:
            trend_direction = "stable"

        # Calculate change percentage
        if older_avg > 0:
            change_percentage = ((recent_avg - older_avg) / older_avg) * 100
        else:
            change_percentage = 0.0

        # Determine significance
        if abs(change_percentage) > 20:
            significance = "critical"
        elif abs(change_percentage) > 10:
            significance = "warning"
        else:
            significance = "info"

        return TrendData(
            metric_name=metric_name,
            historical_values=historical_values,
            current_value=current_value,
            trend_direction=trend_direction,
            change_percentage=change_percentage,
            significance=significance,
        )

    def _generate_trend_insights(
        self, trends: list[TrendData]
    ) -> list[TestInsight]:
        """Generate insights based on trend analysis."""
        insights = []

        for trend in trends:
            if trend["significance"] in ["critical", "warning"]:
                if trend["metric_name"] == "success_rate":
                    if trend["trend_direction"] == "degrading":
                        insights.append(
                            TestInsight(
                                insight_type="regression",
                                title="Test Success Rate Declining",
                                description=(
                                    f"Success rate decreased by "
                                    f"{abs(trend['change_percentage']):.1f}%"
                                ),
                                severity=trend["significance"],
                                recommended_actions=[
                                    "Review recent code changes for "
                                    "test-breaking modifications",
                                    "Check for infrastructure or "
                                    "dependency issues",
                                    "Analyze failure patterns to "
                                    "identify root causes",
                                ],
                                affected_tests=[],
                            )
                        )
                    elif trend["trend_direction"] == "improving":
                        insights.append(
                            TestInsight(
                                insight_type="improvement",
                                title="Test Success Rate Improving",
                                description=(
                                    f"Success rate improved by "
                                    f"{trend['change_percentage']:.1f}%"
                                ),
                                severity="info",
                                recommended_actions=[
                                    "Document changes that led to improvement",
                                    "Consider applying similar improvements "
                                    "to other test areas",
                                ],
                                affected_tests=[],
                            )
                        )

                elif trend["metric_name"] == "total_duration":
                    if trend["trend_direction"] == "degrading":
                        insights.append(
                            TestInsight(
                                insight_type="performance",
                                title="Test Execution Time Increasing",
                                description=(
                                    f"Execution time increased by "
                                    f"{trend['change_percentage']:.1f}%"
                                ),
                                severity=trend["significance"],
                                recommended_actions=[
                                    "Profile slow tests to identify "
                                    "bottlenecks",
                                    "Consider parallel execution optimization",
                                    "Review for unnecessary waits or "
                                    "inefficient selectors",
                                ],
                                affected_tests=[],
                            )
                        )

        return insights

    def _detect_regressions(
        self, trends: list[TrendData]
    ) -> list[dict[str, Any]]:
        """Detect potential regressions based on trends."""
        regression_alerts = []

        for trend in trends:
            if (
                trend["significance"] == "critical"
                and trend["trend_direction"] == "degrading"
            ):
                regression_alerts.append(
                    {
                        "metric": trend["metric_name"],
                        "severity": "high",
                        "change": f"{trend['change_percentage']:.1f}%",
                        "description": (
                            f"Critical regression in {trend['metric_name']}"
                        ),
                        "recommended_actions": [
                            "Investigate recent changes immediately",
                            "Consider rolling back recent deployments",
                            "Escalate to development team",
                        ],
                    }
                )

        return regression_alerts

    def store_execution_summary(self, summary: dict[str, Any]) -> None:
        """Store execution summary for historical trend analysis."""
        filename = (
            self.historical_data_path / f"execution_summary_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.debug(f"Stored execution summary: {filename}")
