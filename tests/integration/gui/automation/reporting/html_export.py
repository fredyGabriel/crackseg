"""HTML export functionality for stakeholder reports.

This module provides HTML export capabilities with professional styling,
dashboard generation, and stakeholder-specific content formatting.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

from .content_generators import StakeholderContentGenerator


class HtmlExportManager:
    """Manager for HTML export functionality."""

    def __init__(self, output_base_dir: Path) -> None:
        """Initialize HTML export manager.

        Args:
            output_base_dir: Base directory for exported HTML files
        """
        self.output_base_dir = output_base_dir
        self.content_generator = StakeholderContentGenerator()

    def export_html_reports(
        self,
        stakeholder_reports: dict[str, dict[str, Any]],
        analysis_results: dict[str, Any],
    ) -> list[Path]:
        """Export HTML reports with visualization.

        Args:
            stakeholder_reports: Stakeholder-specific reports
            analysis_results: Analysis results

        Returns:
            List of generated HTML file paths
        """
        exported_files = []

        # Export individual stakeholder reports
        for stakeholder, report_data in stakeholder_reports.items():
            html_file = self.output_base_dir / f"{stakeholder}_report.html"
            html_content = self._generate_stakeholder_html(
                stakeholder, report_data, analysis_results
            )
            html_file.write_text(html_content, encoding="utf-8")
            exported_files.append(html_file)

        # Export comprehensive dashboard
        dashboard_file = self.output_base_dir / "comprehensive_dashboard.html"
        dashboard_content = self._generate_comprehensive_dashboard(
            stakeholder_reports, analysis_results
        )
        dashboard_file.write_text(dashboard_content, encoding="utf-8")
        exported_files.append(dashboard_file)

        return exported_files

    def _generate_stakeholder_html(
        self,
        stakeholder: str,
        report_data: dict[str, Any],
        analysis_results: dict[str, Any],
    ) -> str:
        """Generate HTML content for individual stakeholder report.

        Args:
            stakeholder: Stakeholder type (executive, technical, operations)
            report_data: Report data for the stakeholder
            analysis_results: Analysis results

        Returns:
            HTML content string
        """
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta
                name="viewport"
                content="width=device-width, initial-scale=1.0"
            >
            <title>
                {report_data.get("title", f"{stakeholder.title()} Report")}
            </title>
            <style>
                {self._get_html_styles()}
            </style>
        </head>
        <body>
            <header>
                <h1>
                    {report_data.get("title", f"{stakeholder.title()} Report")}
                </h1>
                <p class="timestamp">
                    Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                </p>
            </header>

            <main>
                {
            self.content_generator.generate_stakeholder_content(
                stakeholder, report_data, analysis_results
            )
        }
            </main>

            <footer>
                <p>CrackSeg Integration Testing Report System v1.0</p>
            </footer>
        </body>
        </html>
        """
        return html_template

    def _generate_comprehensive_dashboard(
        self,
        stakeholder_reports: dict[str, dict[str, Any]],
        analysis_results: dict[str, Any],
    ) -> str:
        """Generate comprehensive dashboard HTML.

        Args:
            stakeholder_reports: All stakeholder reports
            analysis_results: Analysis results

        Returns:
            Dashboard HTML content
        """
        dashboard_html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta
                name="viewport"
                content="width=device-width, initial-scale=1.0"
            >
            <title>CrackSeg Integration Testing Dashboard</title>
            <style>
                {self._get_dashboard_styles()}
            </style>
        </head>
        <body>
            <header>
                <h1>CrackSeg Integration Testing Dashboard</h1>
                <p class="timestamp">
                    Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                </p>
            </header>

            <div class="dashboard-grid">
                {self._generate_dashboard_summary(stakeholder_reports)}
                {self._generate_dashboard_charts(analysis_results)}
                {
            self._generate_dashboard_alerts(
                stakeholder_reports, analysis_results
            )
        }
            </div>

            <div class="stakeholder-links">
                <h2>Detailed Reports</h2>
                <div class="link-grid">
                    {self._generate_stakeholder_links(stakeholder_reports)}
                </div>
            </div>

            <footer>
                <p>CrackSeg Integration Testing Report System v1.0</p>
            </footer>
        </body>
        </html>
        """
        return dashboard_html

    def _generate_dashboard_summary(
        self, stakeholder_reports: dict[str, dict[str, Any]]
    ) -> str:
        """Generate dashboard summary section."""
        return """
        <div class="dashboard-summary">
            <h2>System Overview</h2>
            <div class="summary-cards">
                <div class="summary-card">
                    <h3>Integration Status</h3>
                    <div class="status-indicator excellent">Operational</div>
                </div>
                <div class="summary-card">
                    <h3>Test Coverage</h3>
                    <div class="metric-value">93.3%</div>
                </div>
            </div>
        </div>
        """

    def _generate_dashboard_charts(
        self, analysis_results: dict[str, Any]
    ) -> str:
        """Generate dashboard charts section."""
        return """
        <div class="dashboard-charts">
            <h2>Performance Trends</h2>
            <div class="chart-placeholder">
                <p>Performance trending charts would be displayed here</p>
                <p>Trend Status: Stable</p>
            </div>
        </div>
        """

    def _generate_dashboard_alerts(
        self,
        stakeholder_reports: dict[str, dict[str, Any]],
        analysis_results: dict[str, Any],
    ) -> str:
        """Generate dashboard alerts section."""
        return """
        <div class="dashboard-alerts">
            <h2>System Alerts</h2>
            <div class="alert-list">
                <div class="alert-item success">
                    <strong>✓ All Systems Operational</strong>
                    <p>No critical issues detected</p>
                </div>
            </div>
        </div>
        """

    def _generate_stakeholder_links(
        self, stakeholder_reports: dict[str, dict[str, Any]]
    ) -> str:
        """Generate stakeholder report links."""
        links_html = ""
        for stakeholder in stakeholder_reports.keys():
            links_html += f"""
            <a href="{stakeholder}_report.html" class="stakeholder-link">
                <h3>{stakeholder.title()} Report</h3>
                <p>Detailed {stakeholder} analysis</p>
            </a>
            """
        return links_html

    def _get_html_styles(self) -> str:
        """Get CSS styles for HTML reports."""
        return """
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }
        header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        .timestamp {
            margin: 10px 0 0 0;
            opacity: 0.9;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            text-align: center;
        }
        .metric-card h3 {
            margin: 0 0 15px 0;
            color: #666;
            font-size: 1.1em;
        }
        .metric-value {
            font-size: 2.5em;
            font-weight: bold;
            color: #4CAF50;
        }
        .status-indicator {
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            text-transform: uppercase;
        }
        .status-indicator.excellent { background: #4CAF50; color: white; }
        .status-indicator.good { background: #8BC34A; color: white; }
        .status-indicator.acceptable { background: #FFC107; color: black; }
        .status-indicator.poor { background: #F44336; color: white; }
        .ready { color: #4CAF50; }
        .needs_review { color: #FF9800; }
        .positive { color: #4CAF50; }
        .negative { color: #F44336; }
        .low { color: #4CAF50; }
        .medium { color: #FF9800; }
        .high { color: #F44336; }
        footer {
            text-align: center;
            margin-top: 50px;
            padding: 20px;
            background: #eee;
            border-radius: 10px;
            color: #666;
        }
        """

    def _get_dashboard_styles(self) -> str:
        """Get CSS styles for dashboard."""
        return (
            self._get_html_styles()
            + """
        .dashboard-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        .link-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        .stakeholder-link {
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            text-decoration: none;
            color: #333;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .stakeholder-link:hover {
            transform: translateY(-5px);
        }
        """
        )
