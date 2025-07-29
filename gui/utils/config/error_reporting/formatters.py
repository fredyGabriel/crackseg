"""
Formatters for error reports.

This module provides various formatting options for error reports,
including text, HTML, and Markdown formats.
"""

from typing import Any


class ErrorReportFormatter:
    """Formatter for error reports in various formats."""

    def format_report(
        self, report: dict[str, Any], format_type: str = "text"
    ) -> str:
        """
        Format error report in the specified format.

        Args:
            report: Error report dictionary
            format_type: Format type ("text", "html", "markdown")

        Returns:
            Formatted report string
        """
        if format_type == "html":
            return self._format_html_report(report)
        elif format_type == "markdown":
            return self._format_markdown_report(report)
        else:
            return self._format_text_report(report)

    def _format_text_report(self, report: dict[str, Any]) -> str:
        """Format report as plain text."""
        lines = []

        # Header
        lines.append("=" * 80)
        lines.append("CRACKSEG CONFIGURATION VALIDATION REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Summary
        summary = report.get("summary", {})
        status = summary.get("overall_status", "unknown").upper()
        lines.append(f"Overall Status: {status}")
        lines.append(f"Total Errors: {summary.get('total_errors', 0)}")
        lines.append(f"Total Warnings: {summary.get('total_warnings', 0)}")
        lines.append("")

        # Errors
        errors = report.get("errors", {})
        if errors.get("total", 0) > 0:
            lines.append("ERRORS:")
            lines.append("-" * 40)
            for error_report in errors.get("reports", []):
                lines.append(f"• {error_report.title}")
                lines.append(f"  {error_report.description}")
                if error_report.suggestions:
                    lines.append("  Suggestions:")
                    for suggestion in error_report.suggestions:
                        lines.append(f"    - {suggestion}")
                lines.append("")

        # Warnings
        warnings = report.get("warnings", {})
        if warnings.get("total", 0) > 0:
            lines.append("WARNINGS:")
            lines.append("-" * 40)
            for warning_report in warnings.get("reports", []):
                lines.append(f"• {warning_report.title}")
                lines.append(f"  {warning_report.description}")
                if warning_report.suggestions:
                    lines.append("  Suggestions:")
                    for suggestion in warning_report.suggestions:
                        lines.append(f"    - {suggestion}")
                lines.append("")

        # Recommendations
        recommendations = report.get("recommendations", [])
        if recommendations:
            lines.append("RECOMMENDATIONS:")
            lines.append("-" * 40)
            for recommendation in recommendations:
                lines.append(f"• {recommendation}")
            lines.append("")

        # Footer
        lines.append("=" * 80)
        lines.append(
            f"Report generated at: {report.get('timestamp', 'unknown')}"
        )
        if report.get("config_path"):
            lines.append(f"Configuration file: {report['config_path']}")
        lines.append("=" * 80)

        return "\n".join(lines)

    def _format_html_report(self, report: dict[str, Any]) -> str:
        """Format report as HTML."""
        html_lines = []

        # HTML header
        html_lines.append("<!DOCTYPE html>")
        html_lines.append("<html>")
        html_lines.append("<head>")
        html_lines.append(
            "<title>CrackSeg Configuration Validation Report</title>"
        )
        html_lines.append("<style>")
        html_lines.append(
            "body { font-family: Arial, sans-serif; margin: 20px; }"
        )
        html_lines.append(
            ".header { background-color: #f0f0f0; padding: 10px; "
            "border-radius: 5px; }"
        )
        html_lines.append(".error { color: #d32f2f; }")
        html_lines.append(".warning { color: #f57c00; }")
        html_lines.append(".success { color: #388e3c; }")
        html_lines.append(".section { margin: 20px 0; }")
        html_lines.append(
            ".suggestion { background-color: #fff3e0; padding: 5px; "
            "margin: 5px 0; border-left: 3px solid #ff9800; }"
        )
        html_lines.append("</style>")
        html_lines.append("</head>")
        html_lines.append("<body>")

        # Header
        summary = report.get("summary", {})
        status = summary.get("overall_status", "unknown")
        status_class = (
            "success"
            if status == "success"
            else "error" if status == "error" else "warning"
        )

        html_lines.append('<div class="header">')
        html_lines.append("<h1>CrackSeg Configuration Validation Report</h1>")
        html_lines.append(
            f'<p><strong>Status:</strong> <span class="{status_class}">'
            f"{status.upper()}</span></p>"
        )
        html_lines.append(
            f"<p><strong>Errors:</strong> {summary.get('total_errors', 0)} | "
            f"<strong>Warnings:</strong> {summary.get('total_warnings', 0)}"
            "</p>"
        )
        html_lines.append("</div>")

        # Errors
        errors = report.get("errors", {})
        if errors.get("total", 0) > 0:
            html_lines.append('<div class="section">')
            html_lines.append("<h2>Errors</h2>")
            for error_report in errors.get("reports", []):
                html_lines.append('<div class="error">')
                html_lines.append(f"<h3>{error_report.title}</h3>")
                html_lines.append(f"<p>{error_report.description}</p>")
                if error_report.suggestions:
                    html_lines.append("<h4>Suggestions:</h4>")
                    for suggestion in error_report.suggestions:
                        html_lines.append(
                            f'<div class="suggestion">{suggestion}</div>'
                        )
                html_lines.append("</div>")
            html_lines.append("</div>")

        # Warnings
        warnings = report.get("warnings", {})
        if warnings.get("total", 0) > 0:
            html_lines.append('<div class="section">')
            html_lines.append("<h2>Warnings</h2>")
            for warning_report in warnings.get("reports", []):
                html_lines.append('<div class="warning">')
                html_lines.append(f"<h3>{warning_report.title}</h3>")
                html_lines.append(f"<p>{warning_report.description}</p>")
                if warning_report.suggestions:
                    html_lines.append("<h4>Suggestions:</h4>")
                    for suggestion in warning_report.suggestions:
                        html_lines.append(
                            f'<div class="suggestion">{suggestion}</div>'
                        )
                html_lines.append("</div>")
            html_lines.append("</div>")

        # Recommendations
        recommendations = report.get("recommendations", [])
        if recommendations:
            html_lines.append('<div class="section">')
            html_lines.append("<h2>Recommendations</h2>")
            html_lines.append("<ul>")
            for recommendation in recommendations:
                html_lines.append(f"<li>{recommendation}</li>")
            html_lines.append("</ul>")
            html_lines.append("</div>")

        # Footer
        html_lines.append('<div class="section">')
        html_lines.append(
            f"<p><strong>Generated:</strong> "
            f"{report.get('timestamp', 'unknown')}</p>"
        )
        if report.get("config_path"):
            html_lines.append(
                f"<p><strong>Config file:</strong> {report['config_path']}</p>"
            )
        html_lines.append("</div>")

        html_lines.append("</body>")
        html_lines.append("</html>")

        return "\n".join(html_lines)

    def _format_markdown_report(self, report: dict[str, Any]) -> str:
        """Format report as Markdown."""
        lines = []

        # Header
        lines.append("# CrackSeg Configuration Validation Report")
        lines.append("")

        # Summary
        summary = report.get("summary", {})
        status = summary.get("overall_status", "unknown")
        status_emoji = (
            "✅" if status == "success" else "❌" if status == "error" else "⚠️"
        )

        lines.append(f"## Summary {status_emoji}")
        lines.append("")
        lines.append(f"- **Status:** {status.upper()}")
        lines.append(f"- **Errors:** {summary.get('total_errors', 0)}")
        lines.append(f"- **Warnings:** {summary.get('total_warnings', 0)}")
        lines.append("")

        # Errors
        errors = report.get("errors", {})
        if errors.get("total", 0) > 0:
            lines.append("## Errors ❌")
            lines.append("")
            for error_report in errors.get("reports", []):
                lines.append(f"### {error_report.title}")
                lines.append(f"{error_report.description}")
                if error_report.suggestions:
                    lines.append("**Suggestions:**")
                    for suggestion in error_report.suggestions:
                        lines.append(f"- {suggestion}")
                lines.append("")

        # Warnings
        warnings = report.get("warnings", {})
        if warnings.get("total", 0) > 0:
            lines.append("## Warnings ⚠️")
            lines.append("")
            for warning_report in warnings.get("reports", []):
                lines.append(f"### {warning_report.title}")
                lines.append(f"{warning_report.description}")
                if warning_report.suggestions:
                    lines.append("**Suggestions:**")
                    for suggestion in warning_report.suggestions:
                        lines.append(f"- {suggestion}")
                lines.append("")

        # Recommendations
        recommendations = report.get("recommendations", [])
        if recommendations:
            lines.append("## Recommendations 💡")
            lines.append("")
            for recommendation in recommendations:
                lines.append(f"- {recommendation}")
            lines.append("")

        # Footer
        lines.append("---")
        lines.append(f"*Generated at: {report.get('timestamp', 'unknown')}*")
        if report.get("config_path"):
            lines.append(f"*Config file: {report['config_path']}*")

        return "\n".join(lines)
