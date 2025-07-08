"""Visualization module for generating interactive performance charts.

This module provides a unified interface for creating comprehensive chart
visualizations by delegating to specialized chart generators.
"""

from __future__ import annotations

import logging
from typing import Any

from tests.e2e.performance.reporting.comparison_charts import (
    ComparisonChartGenerator,
)
from tests.e2e.performance.reporting.config import CHART_THEMES
from tests.e2e.performance.reporting.summary_charts import (
    SummaryChartGenerator,
)
from tests.e2e.performance.reporting.trend_charts import TrendChartGenerator

logger = logging.getLogger(__name__)


class PerformanceVisualizer:
    """Unified interface for creating interactive performance charts."""

    def __init__(self, theme: str = "plotly_white") -> None:
        """Initialize visualizer with theme configuration."""
        self.theme = CHART_THEMES.get(theme, theme)
        self.logger = logging.getLogger(__name__)

        # Initialize specialized chart generators
        self.summary_generator = SummaryChartGenerator(self.theme)
        self.trend_generator = TrendChartGenerator(self.theme)
        self.comparison_generator = ComparisonChartGenerator(self.theme)

    def create_performance_visualizations(
        self, report_content: dict[str, Any]
    ) -> dict[str, str]:
        """Create comprehensive set of performance visualizations."""
        visualizations = {}

        try:
            # Performance summary chart
            summary_fig = (
                self.summary_generator.create_performance_summary_chart(
                    report_content
                )
            )
            vis_key = "performance_summary"
            visualizations[vis_key] = summary_fig.to_html(
                include_plotlyjs="inline", div_id="performance-summary"
            )

            # Trend analysis chart
            if "trend_analysis" in report_content:
                trend_fig = self.trend_generator.create_trend_analysis_chart(
                    report_content["trend_analysis"]
                )
                visualizations["trend_analysis"] = trend_fig.to_html(
                    include_plotlyjs="inline", div_id="trend-analysis"
                )

            # Historical comparison chart
            if "historical_data_summary" in report_content:
                comp_gen = self.comparison_generator
                history_fig = comp_gen.create_historical_comparison_chart(
                    report_content
                )
                vis_key = "historical_comparison"
                visualizations[vis_key] = history_fig.to_html(
                    include_plotlyjs="inline", div_id="historical-comparison"
                )

            # Resource utilization chart
            resource_fig = (
                self.summary_generator.create_resource_utilization_chart(
                    report_content
                )
            )
            vis_key = "resource_utilization"
            visualizations[vis_key] = resource_fig.to_html(
                include_plotlyjs="inline", div_id="resource-utilization"
            )

            # Regression analysis chart
            if "regression_analysis" in report_content:
                regression_fig = (
                    self.trend_generator.create_regression_analysis_chart(
                        report_content["regression_analysis"]
                    )
                )
                vis_key = "regression_analysis"
                visualizations[vis_key] = regression_fig.to_html(
                    include_plotlyjs="inline", div_id="regression-analysis"
                )

        except Exception as e:
            self.logger.error(f"Error creating visualizations: {e}")
            # Fallback to error chart
            error_fig = self.comparison_generator.create_error_chart(str(e))
            visualizations["error"] = error_fig.to_html(
                include_plotlyjs="inline", div_id="error-chart"
            )

        return visualizations

    def create_summary_chart(self, report_content: dict[str, Any]) -> str:
        """Create only the performance summary chart."""
        try:
            fig = self.summary_generator.create_performance_summary_chart(
                report_content
            )
            return fig.to_html(
                include_plotlyjs="inline", div_id="summary-chart"
            )
        except Exception as e:
            self.logger.error(f"Error creating summary chart: {e}")
            error_fig = self.comparison_generator.create_error_chart(str(e))
            return error_fig.to_html(
                include_plotlyjs="inline", div_id="error-chart"
            )

    def create_trend_chart(self, trend_data: dict[str, Any]) -> str:
        """Create only the trend analysis chart."""
        try:
            fig = self.trend_generator.create_trend_analysis_chart(trend_data)
            return fig.to_html(include_plotlyjs="inline", div_id="trend-chart")
        except Exception as e:
            self.logger.error(f"Error creating trend chart: {e}")
            error_fig = self.trend_generator.create_no_trend_data_chart(
                "Error creating trend chart"
            )
            return error_fig.to_html(
                include_plotlyjs="inline", div_id="error-chart"
            )

    def create_comparison_chart(self, report_content: dict[str, Any]) -> str:
        """Create only the historical comparison chart."""
        try:
            comp_gen = self.comparison_generator
            fig = comp_gen.create_historical_comparison_chart(report_content)
            return fig.to_html(
                include_plotlyjs="inline", div_id="comparison-chart"
            )
        except Exception as e:
            self.logger.error(f"Error creating comparison chart: {e}")
            error_fig = self.comparison_generator.create_error_chart(
                "Error creating comparison chart"
            )
            return error_fig.to_html(
                include_plotlyjs="inline", div_id="error-chart"
            )
