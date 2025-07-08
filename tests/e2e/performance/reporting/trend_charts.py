"""Trend analysis chart generation module.

This module handles the creation of trend analysis charts for performance
metrics over time.
"""

from __future__ import annotations

import logging
from typing import Any

import plotly.graph_objects as go

logger = logging.getLogger(__name__)


class TrendChartGenerator:
    """Handles creation of trend analysis charts."""

    def __init__(self, theme: str = "plotly_white") -> None:
        """Initialize trend chart generator with theme."""
        self.theme = theme
        self.logger = logging.getLogger(__name__)

    def create_trend_analysis_chart(
        self, trend_data: dict[str, Any]
    ) -> go.Figure:
        """Create trend analysis chart from trend data."""
        fig = go.Figure()

        # Add trend lines for each metric
        for metric_name, metric_data in trend_data.items():
            if isinstance(metric_data, dict) and "trend" in metric_data:
                trend_values = metric_data["trend"]
                confidence = metric_data.get("confidence", 0.5)

                # Create time series data (simulated)
                time_points = [
                    f"T-{i}" for i in range(len(trend_values), 0, -1)
                ]

                # Color based on trend direction
                color = self._get_trend_color(metric_name, trend_values)

                fig.add_trace(
                    go.Scatter(
                        x=time_points,
                        y=trend_values,
                        mode="lines+markers",
                        name=f"{metric_name} (conf: {confidence:.2f})",
                        line={"color": color, "width": 3},
                        marker={"size": 8},
                    )
                )

        fig.update_layout(
            template=self.theme,
            title="Performance Trends Over Time",
            xaxis_title="Time Period",
            yaxis_title="Performance Metric Value",
            height=400,
            hovermode="x unified",
        )

        return fig

    def create_regression_analysis_chart(
        self, regression_data: dict[str, Any]
    ) -> go.Figure:
        """Create regression analysis chart."""
        fig = go.Figure()

        detected_regressions = regression_data.get("detected_regressions", [])
        performance_improvements = regression_data.get(
            "performance_improvements", []
        )

        if detected_regressions:
            regression_metrics = [
                r.get("metric", "Unknown") for r in detected_regressions
            ]
            regression_impacts = [
                r.get("impact_percentage", 0) for r in detected_regressions
            ]

            fig.add_trace(
                go.Bar(
                    x=regression_metrics,
                    y=regression_impacts,
                    name="Regressions",
                    marker_color="red",
                    text=[f"{impact:.1f}%" for impact in regression_impacts],
                    textposition="auto",
                )
            )

        if performance_improvements:
            improvement_metrics = [
                i.get("metric", "Unknown") for i in performance_improvements
            ]
            improvement_impacts = [
                i.get("impact_percentage", 0) for i in performance_improvements
            ]

            fig.add_trace(
                go.Bar(
                    x=improvement_metrics,
                    y=improvement_impacts,
                    name="Improvements",
                    marker_color="green",
                    text=[f"+{impact:.1f}%" for impact in improvement_impacts],
                    textposition="auto",
                )
            )

        fig.update_layout(
            template=self.theme,
            title="Performance Regression Analysis",
            xaxis_title="Metrics",
            yaxis_title="Impact (%)",
            height=400,
            barmode="group",
        )

        return fig

    def _get_trend_color(
        self, metric_name: str, trend_values: list[float]
    ) -> str:
        """Get color based on metric type and trend direction."""
        if not trend_values or len(trend_values) < 2:
            return "gray"

        # Calculate trend direction
        trend_direction = trend_values[-1] - trend_values[0]

        # Color based on whether increase/decrease is good for this metric
        if metric_name.lower() in ["success_rate", "throughput"]:
            # Higher is better
            return "green" if trend_direction >= 0 else "red"
        elif metric_name.lower() in [
            "violations",
            "memory_usage",
            "cpu_usage",
        ]:
            # Lower is better
            return "red" if trend_direction >= 0 else "green"
        else:
            # Neutral color for unknown metrics
            return "blue"

    def create_no_trend_data_chart(self, message: str) -> go.Figure:
        """Create chart when no trend data is available."""
        fig = go.Figure()

        fig.add_annotation(
            x=0.5,
            y=0.5,
            text=message,
            showarrow=False,
            font={"size": 16},
            xref="paper",
            yref="paper",
            align="center",
        )

        fig.update_layout(
            template=self.theme,
            title="Trend Analysis",
            xaxis={"visible": False},
            yaxis={"visible": False},
            height=300,
        )

        return fig
