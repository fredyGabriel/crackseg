"""Multi-format export manager for stakeholder reports.

This module provides the main orchestration for exporting stakeholder reports
in multiple formats (HTML, JSON, CSV) using specialized export modules.
"""

from pathlib import Path
from typing import Any

from .csv_export import CsvExportManager
from .html_export import HtmlExportManager
from .json_export import JsonExportManager


class MultiFormatExportManager:
    """Manager for exporting reports in multiple formats."""

    def __init__(self, output_base_dir: Path | None = None) -> None:
        """Initialize multi-format export manager.

        Args:
            output_base_dir: Base directory for exported reports
        """
        self.output_base_dir = output_base_dir or Path("comprehensive_reports")
        self.output_base_dir.mkdir(parents=True, exist_ok=True)

        # Initialize specialized export managers
        self.html_manager = HtmlExportManager(self.output_base_dir)
        self.json_manager = JsonExportManager(self.output_base_dir)
        self.csv_manager = CsvExportManager(self.output_base_dir)

    def export_stakeholder_reports(
        self,
        stakeholder_reports: dict[str, dict[str, Any]],
        analysis_results: dict[str, Any],
        export_formats: list[str],
    ) -> list[Path]:
        """Export stakeholder reports in specified formats.

        Args:
            stakeholder_reports: Stakeholder-specific reports
            analysis_results: Analysis results from trend/regression analysis
            export_formats: List of export formats (html, json, csv)

        Returns:
            List of generated export file paths
        """
        exported_files = []

        for export_format in export_formats:
            if export_format.lower() == "html":
                exported_files.extend(
                    self.html_manager.export_html_reports(
                        stakeholder_reports, analysis_results
                    )
                )
            elif export_format.lower() == "json":
                exported_files.extend(
                    self.json_manager.export_json_reports(
                        stakeholder_reports, analysis_results
                    )
                )
            elif export_format.lower() == "csv":
                exported_files.extend(
                    self.csv_manager.export_csv_reports(
                        stakeholder_reports, analysis_results
                    )
                )

        return exported_files
