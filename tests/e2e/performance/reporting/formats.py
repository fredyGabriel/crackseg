"""Report format exporters for multiple output types.

This module provides a unified interface for generating reports in various
formats by delegating to specialized formatter classes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from tests.e2e.performance.reporting.html_formatter import HtmlFormatter
from tests.e2e.performance.reporting.json_formatter import JsonFormatter
from tests.e2e.performance.reporting.pdf_formatter import PdfFormatter

logger = logging.getLogger(__name__)


class ReportFormatter:
    """Unified interface for formatting reports in multiple output formats."""

    def __init__(self, storage_path: Path) -> None:
        """Initialize report formatter with storage configuration."""
        self.storage_path = storage_path
        self.logger = logging.getLogger(__name__)

        # Initialize specialized formatters
        self.html_formatter = HtmlFormatter(storage_path)
        self.json_formatter = JsonFormatter(storage_path)
        self.pdf_formatter = PdfFormatter(storage_path)

    def generate_html_dashboard(
        self, report_content: dict[str, Any], visualizations: dict[str, str]
    ) -> Path:
        """
        Generate comprehensive HTML dashboard with embedded visualizations.
        """
        return self.html_formatter.generate_html_dashboard(
            report_content, visualizations
        )

    def generate_json_report(self, report_content: dict[str, Any]) -> Path:
        """Generate JSON format report for programmatic access."""
        return self.json_formatter.generate_json_report(report_content)

    def generate_pdf_summary(self, report_content: dict[str, Any]) -> Path:
        """Generate PDF summary report."""
        return self.pdf_formatter.generate_pdf_summary(report_content)

    def generate_all_formats(
        self, report_content: dict[str, Any], visualizations: dict[str, str]
    ) -> dict[str, Path]:
        """Generate reports in all available formats."""
        results = {}

        try:
            results["html"] = self.generate_html_dashboard(
                report_content, visualizations
            )
        except Exception as e:
            self.logger.error(f"Failed to generate HTML dashboard: {e}")

        try:
            results["json"] = self.generate_json_report(report_content)
        except Exception as e:
            self.logger.error(f"Failed to generate JSON report: {e}")

        try:
            results["pdf"] = self.generate_pdf_summary(report_content)
        except Exception as e:
            self.logger.error(f"Failed to generate PDF summary: {e}")

        return results

    def get_executive_summary(self, report_content: dict[str, Any]) -> str:
        """Get executive summary for quick overview."""
        return self.pdf_formatter.generate_executive_summary(report_content)

    def get_structured_data(
        self, report_content: dict[str, Any]
    ) -> dict[str, Any]:
        """Get structured data format for API responses."""
        return self.json_formatter.generate_structured_data(report_content)
