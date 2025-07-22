"""
CSV export functionality for stakeholder reports. This module provides
CSV export capabilities for spreadsheet analysis and data processing
workflows.
"""

import csv
from pathlib import Path
from typing import Any


class CsvExportManager:
    """Manager for CSV export functionality."""

    def __init__(self, output_base_dir: Path) -> None:
        """
        Initialize CSV export manager. Args: output_base_dir: Base directory
        for exported CSV files
        """
        self.output_base_dir = output_base_dir

    def export_csv_reports(
        self,
        stakeholder_reports: dict[str, dict[str, Any]],
        analysis_results: dict[str, Any],
    ) -> list[Path]:
        """
        Export CSV reports for spreadsheet analysis. Args:
        stakeholder_reports: Stakeholder-specific reports analysis_results:
        Analysis results Returns: List of generated CSV file paths
        """
        exported_files = []

        # Export metrics summary CSV
        metrics_file = self.output_base_dir / "metrics_summary.csv"
        self._export_metrics_csv(stakeholder_reports, metrics_file)
        exported_files.append(metrics_file)

        # Export trend analysis CSV
        if analysis_results.get("trend_analysis"):
            trends_file = self.output_base_dir / "trend_analysis.csv"
            self._export_trends_csv(
                analysis_results["trend_analysis"], trends_file
            )
            exported_files.append(trends_file)

        # Export regression analysis CSV
        if analysis_results.get("regression_detection"):
            regression_file = self.output_base_dir / "regression_analysis.csv"
            self._export_regression_csv(
                analysis_results["regression_detection"], regression_file
            )
            exported_files.append(regression_file)

        return exported_files

    def _export_metrics_csv(
        self, stakeholder_reports: dict[str, dict[str, Any]], csv_file: Path
    ) -> None:
        """Export metrics summary to CSV."""
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Stakeholder", "Metric", "Value", "Category"])

            for stakeholder, report in stakeholder_reports.items():
                # Export key metrics
                key_metrics = report.get("key_metrics", {})
                for metric, value in key_metrics.items():
                    writer.writerow([stakeholder, metric, value, "key_metric"])

                # Export detailed metrics if available (for technical)
                detailed_metrics = report.get("detailed_metrics", {})
                for metric, value in detailed_metrics.items():
                    if isinstance(value, dict):
                        for sub_metric, sub_value in value.items():
                            writer.writerow(
                                [
                                    stakeholder,
                                    f"{metric}.{sub_metric}",
                                    sub_value,
                                    "detailed_metric",
                                ]
                            )
                    else:
                        writer.writerow(
                            [stakeholder, metric, value, "detailed_metric"]
                        )

    def _export_trends_csv(
        self, trend_analysis: dict[str, Any], csv_file: Path
    ) -> None:
        """Export trend analysis to CSV."""
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Trend_Type", "Metric", "Value", "Status"])

            for trend_type, trend_data in trend_analysis.items():
                if isinstance(trend_data, dict):
                    for metric, value in trend_data.items():
                        writer.writerow([trend_type, metric, value, "current"])

    def _export_regression_csv(
        self, regression_analysis: dict[str, Any], csv_file: Path
    ) -> None:
        """Export regression analysis to CSV."""
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["Regression_Type", "Detected", "Severity", "Details"]
            )

            for (
                regression_type,
                regression_data,
            ) in regression_analysis.items():
                if (
                    isinstance(regression_data, dict)
                    and "regression_detected" in regression_data
                ):
                    writer.writerow(
                        [
                            regression_type,
                            regression_data.get("regression_detected", False),
                            regression_data.get("severity", "none"),
                            regression_data.get("reason", ""),
                        ]
                    )
