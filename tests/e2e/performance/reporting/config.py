"""Configuration classes and type definitions for performance reports.

This module defines configuration settings, data types, and constants used
across the performance reporting system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict


class ReportMetrics(TypedDict):
    """Type definition for report metrics data."""

    timestamp: str
    benchmark_name: str
    success_rate: float
    throughput: float
    avg_response_time: float
    peak_memory_mb: float
    cpu_avg: float


class TrendData(TypedDict):
    """Type definition for trend analysis data."""

    metric_name: str
    trend_direction: str  # "improving", "degrading", "stable"
    trend_percentage: float
    confidence: float
    recommendation: str


@dataclass
class ReportConfiguration:
    """Configuration for performance report generation."""

    include_trend_analysis: bool = True
    include_regression_analysis: bool = True
    include_recommendations: bool = True
    time_window_hours: int = 24
    trend_confidence_threshold: float = 0.8
    max_historical_points: int = 50
    chart_theme: str = "plotly_white"
    export_formats: list[str] = field(default_factory=lambda: ["html", "json"])


# Default export format configuration
DEFAULT_EXPORT_FORMATS = ["html", "json"]

# Chart theme options
CHART_THEMES = {
    "light": "plotly_white",
    "dark": "plotly_dark",
    "minimal": "simple_white",
    "modern": "plotly",
}

# Risk assessment thresholds
RISK_THRESHOLDS = {
    "success_rate": {"critical": 90, "high": 95, "medium": 98},
    "violations": {"critical": 5, "high": 2, "medium": 0},
    "regressions": {"critical": 2, "high": 0, "medium": 0},
}

# Trend analysis constants
TREND_MIN_DATA_POINTS = 3
TREND_STABLE_THRESHOLD = 2.0  # Percentage change threshold for "stable"
TREND_CONFIDENCE_MIN = 0.5
