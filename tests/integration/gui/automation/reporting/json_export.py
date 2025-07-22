"""
JSON export functionality for stakeholder reports. This module
provides JSON export capabilities for external analysis tools and
automated processing of test results.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class JsonExportManager:
    """Manager for JSON export functionality."""

    def __init__(self, output_base_dir: Path) -> None:
        """
        Initialize JSON export manager. Args: output_base_dir: Base directory
        for exported JSON files
        """
        self.output_base_dir = output_base_dir

    def export_json_reports(
        self,
        stakeholder_reports: dict[str, dict[str, Any]],
        analysis_results: dict[str, Any],
    ) -> list[Path]:
        """
        Export JSON reports for external analysis. Args: stakeholder_reports:
        Stakeholder-specific reports analysis_results: Analysis results
        Returns: List of generated JSON file paths
        """
        exported_files = []

        # Export individual stakeholder reports
        for stakeholder, report_data in stakeholder_reports.items():
            json_file = self.output_base_dir / f"{stakeholder}_report.json"
            combined_data = {
                "stakeholder": stakeholder,
                "report": report_data,
                "analysis": analysis_results,
                "export_timestamp": datetime.now().isoformat(),
            }
            json_file.write_text(
                json.dumps(combined_data, indent=2), encoding="utf-8"
            )
            exported_files.append(json_file)

        # Export comprehensive data export
        comprehensive_file = self.output_base_dir / "comprehensive_data.json"
        comprehensive_data = {
            "stakeholder_reports": stakeholder_reports,
            "analysis_results": analysis_results,
            "export_metadata": {
                "export_timestamp": datetime.now().isoformat(),
                "report_version": "1.0",
                "data_sources": [
                    "9.1",
                    "9.2",
                    "9.3",
                    "9.4",
                    "9.5",
                    "9.6",
                    "9.7",
                ],
            },
        }
        comprehensive_file.write_text(
            json.dumps(comprehensive_data, indent=2), encoding="utf-8"
        )
        exported_files.append(comprehensive_file)

        return exported_files
