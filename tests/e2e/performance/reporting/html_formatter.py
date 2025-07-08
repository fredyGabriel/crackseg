"""HTML dashboard formatter for performance reports.

This module handles the generation of HTML dashboards with embedded CSS
and interactive visualizations.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class HtmlFormatter:
    """Handles formatting and generation of HTML dashboards."""

    def __init__(self, storage_path: Path) -> None:
        """Initialize HTML formatter with storage configuration."""
        self.storage_path = storage_path
        self.logger = logging.getLogger(__name__)

    def generate_html_dashboard(
        self, report_content: dict[str, Any], visualizations: dict[str, str]
    ) -> Path:
        """
        Generate comprehensive HTML dashboard with embedded visualizations.
        """
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        html_path = (
            self.storage_path / f"performance_dashboard_{timestamp}.html"
        )

        html_content = self._create_html_template(
            report_content, visualizations
        )

        try:
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            self.logger.info(f"HTML dashboard generated: {html_path}")
        except Exception as e:
            self.logger.error(f"Failed to generate HTML dashboard: {e}")
            raise

        return html_path

    def _create_html_template(
        self, report_content: dict[str, Any], visualizations: dict[str, str]
    ) -> str:
        """Create complete HTML template with embedded visualizations."""
        from tests.e2e.performance.reporting.templates import (
            HTML_TEMPLATE,
            generate_css_styles,
        )

        metadata = report_content.get("metadata", {})
        insights = report_content.get("insights_and_recommendations", {})

        # Extract values with safe defaults
        gen_timestamp = metadata.get("generation_timestamp", "Unknown")
        commit_sha = metadata.get("commit_sha", "Unknown")[:8]
        risk_level = insights.get("risk_assessment", "low")
        summary_text = insights.get("summary", "No summary available")

        # Generate dynamic sections
        key_findings = self._generate_key_findings_section(insights)
        recommendations = self._generate_recommendations_section(insights)
        chart_sections = self._generate_chart_sections(visualizations)
        metadata_section = self._generate_metadata_section(report_content)

        # Format the complete HTML template
        html_content = HTML_TEMPLATE.format(
            css_styles=generate_css_styles(),
            gen_timestamp=gen_timestamp,
            commit_sha=commit_sha,
            risk_level=risk_level,
            summary_text=summary_text,
            risk_assessment=insights.get("risk_assessment", "Unknown").title(),
            key_findings_section=key_findings,
            recommendations_section=recommendations,
            chart_sections=chart_sections,
            metadata_section=metadata_section,
        )

        return html_content

    def _generate_key_findings_section(self, insights: dict[str, Any]) -> str:
        """Generate key findings section of the HTML report."""
        findings = insights.get("key_findings", [])
        if not findings:
            return ""

        findings_html = """
        <div class="summary-card">
            <h2>📊 Key Findings</h2>
            <ul>
        """

        for finding in findings:
            findings_html += f"<li>{finding}</li>"

        findings_html += """
            </ul>
        </div>
        """

        return findings_html

    def _generate_recommendations_section(
        self, insights: dict[str, Any]
    ) -> str:
        """Generate recommendations section of the HTML report."""
        recommendations = insights.get("recommendations", [])
        if not recommendations:
            return ""

        recommendations_html = """
        <div class="recommendations">
            <h2>💡 Recommendations</h2>
            <ul>
        """

        for recommendation in recommendations:
            recommendations_html += f"<li>{recommendation}</li>"

        recommendations_html += """
            </ul>
        </div>
        """

        return recommendations_html

    def _generate_chart_sections(self, visualizations: dict[str, str]) -> str:
        """Generate chart sections for the HTML dashboard."""
        charts_html = ""

        chart_titles = {
            "performance_summary": "Performance Overview",
            "trend_analysis": "Trend Analysis",
            "historical_comparison": "Historical Comparison",
            "resource_utilization": "Resource Utilization",
            "regression_analysis": "Regression Analysis",
        }

        for chart_key, chart_html in visualizations.items():
            title = chart_titles.get(
                chart_key, chart_key.replace("_", " ").title()
            )
            charts_html += f"""
            <div class="chart-container">
                <h3>{title}</h3>
                {chart_html}
            </div>
            """

        return charts_html

    def _generate_metadata_section(
        self, report_content: dict[str, Any]
    ) -> str:
        """Generate metadata section of the HTML report."""
        metadata = report_content.get("metadata", {})
        historical_summary = report_content.get("historical_data_summary", {})

        # Extract values with safe defaults
        gen_time = metadata.get("generation_timestamp", "Unknown")
        commit_sha = metadata.get("commit_sha", "Unknown")
        time_window = metadata.get("time_window_hours", "Unknown")
        data_points = metadata.get("data_points_analyzed", "Unknown")
        time_range = historical_summary.get("time_range", "Unknown")

        metadata_html = f"""
        <div class="summary-card metadata">
            <h3>Report Metadata</h3>
            <p><strong>Generated:</strong> {gen_time}</p>
            <p><strong>Commit:</strong> {commit_sha}</p>
            <p><strong>Time Window:</strong> {time_window} hours</p>
            <p><strong>Data Points:</strong> {data_points}</p>
            <p><strong>Time Range:</strong> {time_range}</p>
        </div>
        """

        return metadata_html
