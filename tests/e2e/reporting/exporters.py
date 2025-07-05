"""Report exporters for different output formats and CI/CD integration.

This module provides specialized exporters for generating reports in various
formats including enhanced HTML dashboards, structured JSON reports, and
CI/CD-friendly formats like JUnit XML.
"""

import json
import logging
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class HTMLReportExporter:
    """Enhanced HTML report exporter with interactive dashboard capabilities.

    Generates comprehensive HTML reports with charts, interactive elements,
    and detailed test result presentation.
    """

    def __init__(self) -> None:
        """Initialize the HTML report exporter."""
        logger.debug("HTML report exporter initialized")

    def export(
        self,
        report_data: dict[str, Any],
        output_path: Path,
        include_charts: bool = True,
    ) -> Path:
        """Export comprehensive HTML report with dashboard features.

        Args:
            report_data: Complete report data
            output_path: Output file path
            include_charts: Include performance charts

        Returns:
            Path to generated HTML file
        """
        html_content = self._generate_enhanced_html(
            report_data, include_charts
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"Generated enhanced HTML report: {output_path}")
        return output_path

    def _generate_enhanced_html(
        self, report_data: dict[str, Any], include_charts: bool
    ) -> str:
        """Generate enhanced HTML report with dashboard features."""
        summary = report_data["execution_summary"]

        css_styles = """
        <style>
            body { font-family: Arial, sans-serif; margin: 20px;
                   background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { background: linear-gradient(135deg, #667eea, #764ba2);
                     color: white; padding: 30px; border-radius: 10px;
                     text-center; }
            .section { background: white; padding: 25px; margin: 25px 0;
                      border-radius: 10px;
                      box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .metrics-grid { display: grid;
                           grid-template-columns: repeat(auto-fit,
                           minmax(200px, 1fr));
                           gap: 20px; margin: 25px 0; }
            .metric-card { background: #f8f9fa; padding: 20px;
                          border-radius: 8px;
                          text-align: center;
                          border-left: 4px solid #667eea; }
            .metric-value { font-size: 2em; font-weight: bold;
                           margin-bottom: 5px; }
            .status-passed { color: #28a745; }
            .status-failed { color: #dc3545; }
            .status-skipped { color: #ffc107; }
        </style>
        """

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CrackSeg E2E Test Report</title>
            {css_styles}
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>CrackSeg E2E Test Report</h1>
                    <p><strong>Success Rate:</strong> <span class="{
            "status-passed"
            if summary.get("success_rate", 0) >= 90
            else "status-failed"
        }">
                    {summary.get("success_rate", 0):.1f}%</span></p>
                </div>
                <div class="section">
                    <h2>Test Execution Overview</h2>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-value">
                            {summary.get("total_tests", 0)}</div>
                            <div>Total Tests</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value status-passed">
                            {summary.get("passed", 0)}</div>
                            <div>Passed</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value status-failed">
                            {summary.get("failed", 0)}</div>
                            <div>Failed</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">
                            {summary.get("total_duration", 0):.2f}s</div>
                            <div>Duration</div>
                        </div>
                    </div>
                </div>
                <div class="section">
                    <h2>Test Results</h2>
                    {
            self._generate_test_results_table(
                report_data.get("test_results", [])
            )
        }
                </div>
            </div>
        </body>
        </html>
        """

    def _generate_test_results_table(
        self, test_results: list[dict[str, Any]]
    ) -> str:
        """Generate test results table."""
        if not test_results:
            return "<p>No test results available.</p>"

        html = "<table style='width: 100%; border-collapse: collapse;'>"
        html += (
            "<tr style='background: #f8f9fa;'>"
            "<th>Test</th><th>Status</th><th>Duration</th></tr>"
        )

        for test in test_results:
            status = test.get("status", "unknown")
            status_class = f"status-{status}"
            html += f"""
            <tr style='border-bottom: 1px solid #eee;'>
                <td style='padding: 10px;'>
                {test.get("test_name", "Unknown")}</td>
                <td style='padding: 10px;' class="{status_class}">
                {status.upper()}</td>
                <td style='padding: 10px;'>
                {test.get("duration", 0):.2f}s</td>
            </tr>
            """

        html += "</table>"
        return html


class JSONReportExporter:
    """JSON report exporter for structured data exchange.

    Generates structured JSON reports suitable for programmatic processing
    and integration with external tools.
    """

    def __init__(self) -> None:
        """Initialize the JSON report exporter."""
        logger.debug("JSON report exporter initialized")

    def export(
        self,
        report_data: dict[str, Any],
        output_path: Path,
        pretty_print: bool = True,
    ) -> Path:
        """Export structured JSON report.

        Args:
            report_data: Complete report data
            output_path: Output file path
            pretty_print: Format JSON with indentation

        Returns:
            Path to generated JSON file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            if pretty_print:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            else:
                json.dump(report_data, f, ensure_ascii=False)

        logger.info(f"Generated JSON report: {output_path}")
        return output_path


class CICDReportExporter:
    """CI/CD integration report exporter.

    Generates reports in formats suitable for CI/CD systems including
    JUnit XML for test result integration and metrics files for dashboards.
    """

    def __init__(self) -> None:
        """Initialize the CI/CD report exporter."""
        logger.debug("CI/CD report exporter initialized")

    def export_junit_xml(
        self,
        report_data: dict[str, Any],
        output_path: Path,
    ) -> Path:
        """Export JUnit XML format for CI/CD integration.

        Args:
            report_data: Complete report data
            output_path: Output file path

        Returns:
            Path to generated JUnit XML file
        """
        test_results = report_data.get("test_results", [])
        summary = report_data.get("execution_summary", {})

        # Create root element
        testsuite = ET.Element("testsuite")
        testsuite.set("name", "CrackSeg E2E Tests")
        testsuite.set("tests", str(summary.get("total_tests", 0)))
        testsuite.set("failures", str(summary.get("failed", 0)))
        testsuite.set("errors", str(summary.get("error", 0)))
        testsuite.set("time", str(summary.get("total_duration", 0)))

        # Add individual test cases
        for test in test_results:
            testcase = ET.SubElement(testsuite, "testcase")
            testcase.set("name", test.get("test_name", "Unknown"))
            testcase.set("time", str(test.get("duration", 0)))

            status = test.get("status", "unknown")
            if status == "failed":
                failure = ET.SubElement(testcase, "failure")
                failure.set(
                    "message", test.get("failure_reason", "Test failed")
                )
                failure.text = test.get("error_message", "")
            elif status == "error":
                error = ET.SubElement(testcase, "error")
                error.set("message", test.get("error_message", "Test error"))

        # Write XML file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tree = ET.ElementTree(testsuite)
        tree.write(output_path, encoding="utf-8", xml_declaration=True)

        logger.info(f"Generated JUnit XML report: {output_path}")
        return output_path

    def export_metrics_file(
        self,
        report_data: dict[str, Any],
        output_path: Path,
    ) -> Path:
        """Export metrics file for CI/CD dashboards.

        Args:
            report_data: Complete report data
            output_path: Output file path

        Returns:
            Path to generated metrics file
        """
        summary = report_data.get("execution_summary", {})
        performance_summary = report_data.get("performance_summary", {})

        metrics = {
            "test_execution": {
                "total_tests": summary.get("total_tests", 0),
                "passed_tests": summary.get("passed", 0),
                "failed_tests": summary.get("failed", 0),
                "success_rate": summary.get("success_rate", 0),
                "total_duration": summary.get("total_duration", 0),
            },
            "performance": {
                "average_page_load_time": performance_summary.get(
                    "average_page_load_time", 0
                ),
                "peak_memory_usage": performance_summary.get(
                    "peak_memory_usage", 0
                ),
            },
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "report_version": "1.0",
            },
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Generated CI/CD metrics file: {output_path}")
        return output_path
