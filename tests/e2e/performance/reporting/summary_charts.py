"""Performance summary chart generation module.

This module handles the creation of performance summary charts and gauges
for the main dashboard overview.
"""

from __future__ import annotations

import logging
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class SummaryChartGenerator:
    """Handles creation of performance summary charts and gauges."""

    def __init__(self, theme: str = "plotly_white") -> None:
        """Initialize summary chart generator with theme."""
        self.theme = theme
        self.logger = logging.getLogger(__name__)

    def create_performance_summary_chart(
        self, report_content: dict[str, Any]
    ) -> go.Figure:
        """Create main performance summary chart with multiple metrics."""
        performance_data = report_content.get("performance_summary", {})
        overall_summary = performance_data.get("overall_summary", {})

        # Create subplots for multiple metrics
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Success Rate",
                "Throughput",
                "Resource Usage",
                "Violations",
            ),
            specs=[
                [{"type": "indicator"}, {"type": "indicator"}],
                [{"type": "bar"}, {"type": "bar"}],
            ],
        )

        # Add success rate gauge
        self._add_success_rate_gauge(fig, overall_summary)

        # Add throughput gauge
        self._add_throughput_gauge(fig, overall_summary)

        # Add resource usage bar chart
        self._add_resource_usage_bars(fig, performance_data)

        # Add violations bar chart
        self._add_violations_bars(fig, overall_summary)

        fig.update_layout(
            template=self.theme,
            title="Performance Summary Dashboard",
            height=600,
            showlegend=False,
        )

        return fig

    def _add_success_rate_gauge(
        self, fig: go.Figure, overall_summary: dict[str, Any]
    ) -> None:
        """Add success rate gauge to the summary chart."""
        success_rate = overall_summary.get("average_success_rate", 0)

        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=success_rate,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Success Rate (%)"},
                gauge={
                    "axis": {"range": [None, 100]},
                    "bar": {
                        "color": "green" if success_rate > 95 else "orange"
                    },
                    "steps": [
                        {"range": [0, 90], "color": "lightgray"},
                        {"range": [90, 95], "color": "yellow"},
                        {"range": [95, 100], "color": "lightgreen"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 95,
                    },
                },
            ),
            row=1,
            col=1,
        )

    def _add_throughput_gauge(
        self, fig: go.Figure, overall_summary: dict[str, Any]
    ) -> None:
        """Add throughput gauge to the summary chart."""
        throughput = overall_summary.get("average_throughput", 0)

        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=throughput,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Throughput (ops/sec)"},
                gauge={
                    "axis": {"range": [None, max(200, throughput * 1.2)]},
                    "bar": {"color": "blue"},
                },
            ),
            row=1,
            col=2,
        )

    def _add_resource_usage_bars(
        self, fig: go.Figure, performance_data: dict[str, Any]
    ) -> None:
        """Add resource usage bar chart to the summary chart."""
        resource_summary = performance_data.get("resource_summary", {})
        resources = ["Memory (MB)", "CPU (%)"]
        resource_values = [
            resource_summary.get("peak_memory_mb", 0),
            resource_summary.get("avg_cpu_usage", 0),
        ]

        fig.add_trace(
            go.Bar(
                x=resources,
                y=resource_values,
                name="Resource Usage",
                marker_color=["red", "orange"],
            ),
            row=2,
            col=1,
        )

    def _add_violations_bars(
        self, fig: go.Figure, overall_summary: dict[str, Any]
    ) -> None:
        """Add violations bar chart to the summary chart."""
        violations = overall_summary.get("total_violations", 0)
        benchmarks = overall_summary.get("total_benchmarks", 1)

        fig.add_trace(
            go.Bar(
                x=["Violations", "Success"],
                y=[violations, benchmarks - violations],
                name="Test Results",
                marker_color=["red", "green"],
            ),
            row=2,
            col=2,
        )

    def create_resource_utilization_chart(
        self, report_content: dict[str, Any]
    ) -> go.Figure:
        """Create resource utilization chart."""
        performance_data = report_content.get("performance_summary", {})
        resource_summary = performance_data.get("resource_summary", {})

        fig = go.Figure()

        # Memory usage over time (simulated data)
        fig.add_trace(
            go.Scatter(
                x=["Start", "Mid", "End"],
                y=[
                    resource_summary.get("peak_memory_mb", 0) * 0.7,
                    resource_summary.get("peak_memory_mb", 0),
                    resource_summary.get("peak_memory_mb", 0) * 0.8,
                ],
                mode="lines+markers",
                name="Memory Usage (MB)",
                line={"color": "red"},
            )
        )

        # CPU usage over time (simulated data)
        fig.add_trace(
            go.Scatter(
                x=["Start", "Mid", "End"],
                y=[
                    resource_summary.get("avg_cpu_usage", 0) * 0.8,
                    resource_summary.get("avg_cpu_usage", 0),
                    resource_summary.get("avg_cpu_usage", 0) * 0.9,
                ],
                mode="lines+markers",
                name="CPU Usage (%)",
                line={"color": "orange"},
                yaxis="y2",
            )
        )

        fig.update_layout(
            template=self.theme,
            title="Resource Utilization Over Time",
            xaxis_title="Time",
            yaxis={"title": "Memory (MB)", "side": "left"},
            yaxis2={
                "title": "CPU (%)",
                "overlaying": "y",
                "side": "right",
                "range": [0, 100],
            },
            height=400,
        )

        return fig
