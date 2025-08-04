"""Report formatting for validation reporting."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

from jinja2 import Environment, FileSystemLoader

if TYPE_CHECKING:
    from .config import ValidationReportData


class ReportFormatter:
    """Formatter for validation reports."""

    def __init__(self) -> None:
        """Initialize formatter."""
        # Setup Jinja2 environment for templating
        self.template_env = Environment(
            loader=FileSystemLoader(
                Path(__file__).parent.parent / "templates"
            ),
            autoescape=True,
        )

    def generate_markdown_report(
        self, report_data: "ValidationReportData"
    ) -> str:
        """Generate markdown format report.

        Args:
            report_data: Validation report data

        Returns:
            Markdown report content
        """
        template = self.template_env.get_template("validation_report.md.j2")
        return template.render(report=report_data)

    def generate_html_report(self, report_data: "ValidationReportData") -> str:
        """Generate HTML format report.

        Args:
            report_data: Validation report data

        Returns:
            HTML report content
        """
        template = self.template_env.get_template("validation_report.html.j2")
        return template.render(report=report_data)

    def generate_pdf_report(self, report_data: "ValidationReportData") -> str:
        """Generate PDF format report.

        Args:
            report_data: Validation report data

        Returns:
            PDF report content (placeholder)
        """
        # Convert markdown to PDF using pandoc or similar
        markdown_content = self.generate_markdown_report(report_data)
        # This would require additional PDF generation logic
        return markdown_content  # Placeholder

    def report_data_to_dict(
        self, report_data: "ValidationReportData"
    ) -> dict[str, Any]:
        """Convert report data to dictionary for JSON export.

        Args:
            report_data: Validation report data

        Returns:
            Dictionary representation of report data
        """
        return {
            "artifact_id": report_data.artifact_id,
            "target_environment": report_data.target_environment,
            "target_format": report_data.target_format,
            "validation_timestamp": report_data.validation_timestamp,
            "validation_duration": report_data.validation_duration,
            "performance": {
                "inference_time_ms": report_data.inference_time_ms,
                "memory_usage_mb": report_data.memory_usage_mb,
                "throughput_rps": report_data.throughput_rps,
                "performance_score": report_data.performance_score,
            },
            "security": {
                "security_score": report_data.security_score,
                "vulnerabilities_found": report_data.vulnerabilities_found,
                "security_scan_passed": report_data.security_scan_passed,
            },
            "compatibility": {
                "compatibility_score": report_data.compatibility_score,
                "python_compatible": report_data.python_compatible,
                "dependencies_compatible": report_data.dependencies_compatible,
                "environment_compatible": report_data.environment_compatible,
            },
            "functional_tests": {
                "functional_tests_passed": report_data.functional_tests_passed,
                "test_coverage_percentage": (
                    report_data.test_coverage_percentage
                ),
                "critical_tests_passed": report_data.critical_tests_passed,
            },
            "resource_utilization": {
                "cpu_usage_percent": report_data.cpu_usage_percent,
                "gpu_usage_percent": report_data.gpu_usage_percent,
                "disk_usage_mb": report_data.disk_usage_mb,
                "network_bandwidth_mbps": report_data.network_bandwidth_mbps,
            },
            "deployment_readiness": {
                "deployment_ready": report_data.deployment_ready,
                "risk_level": report_data.risk_level,
                "estimated_deployment_time": (
                    report_data.estimated_deployment_time
                ),
            },
            "recommendations": report_data.recommendations,
            "warnings": report_data.warnings,
            "critical_issues": report_data.critical_issues,
        }
