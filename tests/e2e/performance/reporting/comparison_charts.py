"""Historical comparison chart generation module.

This module handles the creation of historical comparison charts for
performance metrics across different time periods.
"""

from __future__ import annotations

import logging
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class ComparisonChartGenerator:
    """Handles creation of historical comparison charts."""

    def __init__(self, theme: str = "plotly_white") -> None:
        """Initialize comparison chart generator with theme."""
        self.theme = theme
        self.logger = logging.getLogger(__name__)

    def create_historical_comparison_chart(
        self, report_content: dict[str, Any]
    ) -> go.Figure:
        """Create historical comparison chart."""
        historical_data = report_content.get("historical_data_summary", {})
        current_performance = report_content.get("performance_summary", {})

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Success Rate Comparison",
                "Throughput Comparison",
                "Memory Usage Comparison",
                "Violations Comparison",
            ),
        )

        # Extract current and historical metrics
        current_metrics = self._extract_current_metrics(current_performance)
        historical_metrics = self._extract_historical_metrics(historical_data)

        # Add comparison charts
        self._add_success_rate_comparison(
            fig, current_metrics, historical_metrics
        )
        self._add_throughput_comparison(
            fig, current_metrics, historical_metrics
        )
        self._add_memory_comparison(fig, current_metrics, historical_metrics)
        self._add_violations_comparison(
            fig, current_metrics, historical_metrics
        )

        fig.update_layout(
            template=self.theme,
            title="Historical Performance Comparison",
            height=600,
            showlegend=True,
        )

        return fig

    def _extract_current_metrics(
        self, performance_data: dict[str, Any]
    ) -> dict[str, float]:
        """Extract current performance metrics."""
        overall_summary = performance_data.get("overall_summary", {})
        resource_summary = performance_data.get("resource_summary", {})

        return {
            "success_rate": overall_summary.get("average_success_rate", 0),
            "throughput": overall_summary.get("average_throughput", 0),
            "memory_usage": resource_summary.get("peak_memory_mb", 0),
            "violations": overall_summary.get("total_violations", 0),
        }

    def _extract_historical_metrics(
        self, historical_data: dict[str, Any]
    ) -> dict[str, float]:
        """Extract historical performance metrics."""
        return {
            "success_rate": historical_data.get("avg_success_rate", 0),
            "throughput": historical_data.get("avg_throughput", 0),
            "memory_usage": historical_data.get("avg_memory_usage", 0),
            "violations": historical_data.get("avg_violations", 0),
        }

    def _add_success_rate_comparison(
        self,
        fig: go.Figure,
        current: dict[str, float],
        historical: dict[str, float],
    ) -> None:
        """Add success rate comparison chart."""
        fig.add_trace(
            go.Bar(
                x=["Current", "Historical Avg"],
                y=[current["success_rate"], historical["success_rate"]],
                name="Success Rate",
                marker_color=["blue", "lightblue"],
            ),
            row=1,
            col=1,
        )

    def _add_throughput_comparison(
        self,
        fig: go.Figure,
        current: dict[str, float],
        historical: dict[str, float],
    ) -> None:
        """Add throughput comparison chart."""
        fig.add_trace(
            go.Bar(
                x=["Current", "Historical Avg"],
                y=[current["throughput"], historical["throughput"]],
                name="Throughput",
                marker_color=["green", "lightgreen"],
            ),
            row=1,
            col=2,
        )

    def _add_memory_comparison(
        self,
        fig: go.Figure,
        current: dict[str, float],
        historical: dict[str, float],
    ) -> None:
        """Add memory usage comparison chart."""
        fig.add_trace(
            go.Bar(
                x=["Current", "Historical Avg"],
                y=[current["memory_usage"], historical["memory_usage"]],
                name="Memory Usage",
                marker_color=["red", "lightcoral"],
            ),
            row=2,
            col=1,
        )

    def _add_violations_comparison(
        self,
        fig: go.Figure,
        current: dict[str, float],
        historical: dict[str, float],
    ) -> None:
        """Add violations comparison chart."""
        fig.add_trace(
            go.Bar(
                x=["Current", "Historical Avg"],
                y=[current["violations"], historical["violations"]],
                name="Violations",
                marker_color=["orange", "peachpuff"],
            ),
            row=2,
            col=2,
        )

    def create_no_data_chart(self, message: str) -> go.Figure:
        """Create chart when no historical data is available."""
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
            title="Historical Comparison",
            xaxis={"visible": False},
            yaxis={"visible": False},
            height=300,
        )

        return fig

    def create_error_chart(self, error_message: str) -> go.Figure:
        """Create error chart when chart generation fails."""
        fig = go.Figure()

        fig.add_annotation(
            x=0.5,
            y=0.5,
            text=f"Error generating chart: {error_message}",
            showarrow=False,
            font={"size": 14, "color": "red"},
            xref="paper",
            yref="paper",
            align="center",
        )

        fig.update_layout(
            template=self.theme,
            title="Chart Generation Error",
            xaxis={"visible": False},
            yaxis={"visible": False},
            height=300,
        )

        return fig
