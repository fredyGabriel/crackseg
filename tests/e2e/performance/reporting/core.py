"""Core performance report generator orchestrator.

This module contains the main PerformanceReportGenerator class that coordinates
the entire reporting process using specialized modules for data processing,
analysis, visualization, and formatting.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tests.e2e.performance.metrics_collector import MetricsCollector
from tests.e2e.performance.reporting.analysis import PerformanceAnalyzer
from tests.e2e.performance.reporting.config import ReportConfiguration
from tests.e2e.performance.reporting.data_processor import (
    BenchmarkDataProcessor,
)
from tests.e2e.performance.reporting.formats import ReportFormatter
from tests.e2e.performance.reporting.historical_manager import (
    HistoricalDataManager,
)
from tests.e2e.performance.reporting.visualizations import (
    PerformanceVisualizer,
)

logger = logging.getLogger(__name__)


class PerformanceReportGenerator:
    """
    Main performance report generator orchestrating the complete reporting
    process.
    """

    def __init__(
        self,
        storage_path: str = "performance-reports",
        historical_data_path: str = "performance-historical-data",
        config: ReportConfiguration | None = None,
    ) -> None:
        """
        Initialize performance report generator with comprehensive
        configuration.
        """
        self.storage_path = Path(storage_path)
        self.historical_data_path = Path(historical_data_path)
        self.config = config or ReportConfiguration()

        # Ensure directories exist
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.historical_data_path.mkdir(parents=True, exist_ok=True)

        # Initialize specialized components
        self.metrics_collector = MetricsCollector(self.historical_data_path)
        self.data_processor = BenchmarkDataProcessor()
        self.historical_manager = HistoricalDataManager(
            self.historical_data_path
        )
        self.analyzer = PerformanceAnalyzer(self.config)
        self.visualizer = PerformanceVisualizer(self.config.chart_theme)
        self.formatter = ReportFormatter(self.storage_path)

        self.logger = logging.getLogger(__name__)

    def generate_comprehensive_report(
        self, benchmark_results: dict[str, Any], commit_sha: str | None = None
    ) -> dict[str, Path]:
        """
        Generate comprehensive performance report with multiple output formats.
        """
        self.logger.info(
            "Starting comprehensive performance report generation"
        )

        try:
            # Process performance data
            processed_data = self.data_processor.process_benchmark_results(
                benchmark_results
            )

            # Load historical data for comparative analysis
            historical_data = self.historical_manager.load_historical_data()

            # Generate comprehensive analysis
            analysis_results = self._perform_comprehensive_analysis(
                processed_data, historical_data
            )

            # Create report content structure
            report_content = self._create_report_content(
                processed_data, analysis_results, commit_sha, historical_data
            )

            # Generate visualizations
            visualizations = self._generate_visualizations(report_content)

            # Export to requested formats
            output_files = self._export_reports(report_content, visualizations)

            # Store current data for future historical analysis
            self.historical_manager.store_current_data(
                processed_data, commit_sha
            )

            self.logger.info(
                f"Report generation completed successfully. "
                f"Formats: {list(output_files.keys())}"
            )
            return output_files

        except Exception as e:
            self.logger.error(f"Failed to generate comprehensive report: {e}")
            raise

    def _perform_comprehensive_analysis(
        self,
        current_data: dict[str, Any],
        historical_data: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Perform comprehensive performance analysis including trends and
        regressions.
        """
        analysis_results = {}

        # Trend analysis
        if self.config.include_trend_analysis:
            trend_analysis = self.analyzer.perform_trend_analysis(
                current_data, historical_data
            )
            analysis_results["trend_analysis"] = trend_analysis

        # Regression analysis
        if self.config.include_regression_analysis:
            regression_analysis = self.analyzer.perform_regression_analysis(
                current_data, historical_data
            )
            analysis_results["regression_analysis"] = regression_analysis

        # Generate insights and recommendations
        if self.config.include_recommendations:
            insights = self.analyzer.generate_insights(
                current_data,
                analysis_results.get("trend_analysis", {}),
                analysis_results.get("regression_analysis", {}),
            )
            analysis_results["insights"] = insights

        return analysis_results

    def _create_report_content(
        self,
        processed_data: dict[str, Any],
        analysis_results: dict[str, Any],
        commit_sha: str | None,
        historical_data: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Create comprehensive report content structure."""
        return {
            "metadata": {
                "generation_timestamp": datetime.now(UTC).isoformat(),
                "commit_sha": commit_sha or "unknown",
                "time_window_hours": self.config.time_window_hours,
                "data_points_analyzed": len(historical_data),
            },
            "performance_summary": processed_data,
            "trend_analysis": analysis_results.get("trend_analysis", {}),
            "regression_analysis": analysis_results.get(
                "regression_analysis", {}
            ),
            "insights_and_recommendations": analysis_results.get(
                "insights", {}
            ),
            "historical_data_summary": (
                self.historical_manager.summarize_historical_data(
                    historical_data
                )
            ),
            "historical_data_info": (
                self.historical_manager.get_data_file_info()
            ),
        }

    def _generate_visualizations(
        self, report_content: dict[str, Any]
    ) -> dict[str, str]:
        """Generate visualizations if HTML format is requested."""
        visualizations = {}

        if "html" in self.config.export_formats:
            try:
                visualizations = (
                    self.visualizer.create_performance_visualizations(
                        report_content
                    )
                )
            except Exception as e:
                self.logger.error(f"Failed to generate visualizations: {e}")
                # Continue without visualizations

        return visualizations

    def _export_reports(
        self, report_content: dict[str, Any], visualizations: dict[str, str]
    ) -> dict[str, Path]:
        """Export reports in all configured formats."""
        output_files: dict[str, Path] = {}

        for format_type in self.config.export_formats:
            try:
                if format_type == "html":
                    html_path = self.formatter.generate_html_dashboard(
                        report_content, visualizations
                    )
                    output_files["html"] = html_path

                elif format_type == "json":
                    json_path = self.formatter.generate_json_report(
                        report_content
                    )
                    output_files["json"] = json_path

                elif format_type == "pdf":
                    pdf_path = self.formatter.generate_pdf_summary(
                        report_content
                    )
                    output_files["pdf"] = pdf_path

                else:
                    self.logger.warning(
                        f"Unsupported export format: {format_type}"
                    )

            except Exception as e:
                self.logger.error(
                    f"Failed to export {format_type} format: {e}"
                )

        return output_files

    def get_storage_info(self) -> dict[str, Any]:
        """Get information about storage directories and files."""
        return {
            "storage_path": str(self.storage_path),
            "historical_data_path": str(self.historical_data_path),
            "storage_exists": self.storage_path.exists(),
            "historical_exists": self.historical_data_path.exists(),
            "historical_data_info": (
                self.historical_manager.get_data_file_info()
            ),
        }

    def cleanup_old_data(self, max_files: int = 100) -> int:
        """Clean up old historical data files."""
        return self.historical_manager.cleanup_old_data(max_files)

    def validate_configuration(self) -> dict[str, Any]:
        """Validate the current configuration."""
        validation: dict[str, Any] = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
        }

        # Check export formats
        valid_formats = ["html", "json", "pdf"]
        for fmt in self.config.export_formats:
            if fmt not in valid_formats:
                validation["errors"].append(f"Invalid export format: {fmt}")
                validation["is_valid"] = False

        # Check theme
        valid_themes = [
            "plotly",
            "plotly_white",
            "plotly_dark",
            "ggplot2",
            "seaborn",
        ]
        if self.config.chart_theme not in valid_themes:
            validation["warnings"].append(
                f"Unknown chart theme: {self.config.chart_theme}"
            )

        # Check time window
        if self.config.time_window_hours <= 0:
            validation["errors"].append("Time window must be positive")
            validation["is_valid"] = False

        # Check directory permissions
        if not self.storage_path.exists():
            try:
                self.storage_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                validation["errors"].append(
                    f"Cannot create storage directory: {e}"
                )
                validation["is_valid"] = False

        return validation
