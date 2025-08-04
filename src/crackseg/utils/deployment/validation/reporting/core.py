"""Core validation reporter implementation."""

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import ValidationReportData
from .formatters import ReportFormatter
from .risk_analyzer import RiskAnalyzer
from .visualizations import ChartGenerator

if TYPE_CHECKING:
    from .config import DeploymentConfig


class ValidationReporter:
    """Advanced validation reporting system."""

    def __init__(self, output_dir: Path | None = None) -> None:
        """Initialize validation reporter.

        Args:
            output_dir: Directory to save reports.
            Defaults to artifacts/validation_reports/
        """
        self.output_dir = output_dir or Path("artifacts/validation_reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.formatter = ReportFormatter()
        self.risk_analyzer = RiskAnalyzer()
        self.chart_generator = ChartGenerator()

        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"ValidationReporter initialized with output_dir: "
            f"{self.output_dir}"
        )

    def generate_comprehensive_report(
        self,
        validation_results: dict[str, Any],
        config: "DeploymentConfig",
        detailed_metrics: dict[str, Any] | None = None,
    ) -> ValidationReportData:
        """Generate comprehensive validation report.

        Args:
            validation_results: Results from validation pipeline
            config: Deployment configuration
            detailed_metrics: Additional detailed metrics

        Returns:
            Comprehensive validation report data
        """
        self.logger.info(
            f"Generating comprehensive report for {config.artifact_id}"
        )

        # Calculate risk level
        risk_level = self.risk_analyzer.calculate_risk_level(
            validation_results
        )

        # Generate recommendations
        recommendations = self.risk_analyzer.generate_recommendations(
            validation_results, config
        )
        warnings = self.risk_analyzer.generate_warnings(validation_results)
        critical_issues = self.risk_analyzer.generate_critical_issues(
            validation_results
        )

        # Estimate deployment time
        deployment_time = self.risk_analyzer.estimate_deployment_time(
            validation_results, config
        )

        # Create report data
        report_data = ValidationReportData(
            artifact_id=config.artifact_id,
            target_environment=config.target_environment,
            target_format=config.target_format,
            validation_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            validation_duration=validation_results.get(
                "validation_duration", 0.0
            ),
            # Performance metrics
            inference_time_ms=validation_results.get("inference_time_ms", 0.0),
            memory_usage_mb=validation_results.get("memory_usage_mb", 0.0),
            throughput_rps=validation_results.get(
                "throughput_requests_per_second", 0.0
            ),
            performance_score=validation_results.get("performance_score", 0.0),
            # Security metrics
            security_score=validation_results.get("security_score", 0.0),
            vulnerabilities_found=validation_results.get(
                "vulnerabilities_found", 0
            ),
            security_scan_passed=validation_results.get(
                "security_scan_passed", False
            ),
            # Compatibility metrics
            compatibility_score=validation_results.get(
                "compatibility_score", 0.0
            ),
            python_compatible=validation_results.get(
                "python_compatible", False
            ),
            dependencies_compatible=validation_results.get(
                "dependencies_compatible", False
            ),
            environment_compatible=validation_results.get(
                "environment_compatible", False
            ),
            # Functional test results
            functional_tests_passed=validation_results.get(
                "functional_tests_passed", False
            ),
            test_coverage_percentage=validation_results.get(
                "test_coverage_percentage", 0.0
            ),
            critical_tests_passed=validation_results.get(
                "critical_tests_passed", False
            ),
            # Resource utilization (from detailed metrics)
            cpu_usage_percent=(
                detailed_metrics.get("cpu_usage_percent", 0.0)
                if detailed_metrics
                else 0.0
            ),
            gpu_usage_percent=(
                detailed_metrics.get("gpu_usage_percent", 0.0)
                if detailed_metrics
                else 0.0
            ),
            disk_usage_mb=(
                detailed_metrics.get("disk_usage_mb", 0.0)
                if detailed_metrics
                else 0.0
            ),
            network_bandwidth_mbps=(
                detailed_metrics.get("network_bandwidth_mbps", 0.0)
                if detailed_metrics
                else 0.0
            ),
            # Deployment readiness
            deployment_ready=validation_results.get("success", False),
            risk_level=risk_level,
            estimated_deployment_time=deployment_time,
            # Recommendations
            recommendations=recommendations,
            warnings=warnings,
            critical_issues=critical_issues,
        )

        return report_data

    def save_report(
        self, report_data: ValidationReportData, format: str = "markdown"
    ) -> Path:
        """Save validation report to file.

        Args:
            report_data: Validation report data
            format: Output format ("markdown", "html", "json", "pdf")

        Returns:
            Path to saved report file
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"validation_report_{report_data.artifact_id}_{timestamp}"

        if format == "markdown":
            report_path = self.output_dir / f"{filename}.md"
            content = self.formatter.generate_markdown_report(report_data)
        elif format == "html":
            report_path = self.output_dir / f"{filename}.html"
            content = self.formatter.generate_html_report(report_data)
        elif format == "json":
            report_path = self.output_dir / f"{filename}.json"
            content = json.dumps(
                self.formatter.report_data_to_dict(report_data), indent=2
            )
        elif format == "pdf":
            report_path = self.output_dir / f"{filename}.pdf"
            content = self.formatter.generate_pdf_report(report_data)
        else:
            raise ValueError(f"Unsupported format: {format}")

        report_path.write_text(content)
        self.logger.info(f"Report saved to: {report_path}")

        return report_path

    def generate_performance_charts(
        self,
        report_data: ValidationReportData,
        save_dir: Path | None = None,
    ) -> list[Path]:
        """Generate performance visualization charts.

        Args:
            report_data: Validation report data
            save_dir: Directory to save charts

        Returns:
            List of paths to generated chart files
        """
        save_dir = save_dir or self.output_dir / "charts"
        save_dir.mkdir(parents=True, exist_ok=True)

        charts = []

        # 1. Performance metrics radar chart
        radar_chart = self.chart_generator.create_performance_radar_chart(
            report_data, save_dir
        )
        charts.append(radar_chart)

        # 2. Resource utilization bar chart
        resource_chart = (
            self.chart_generator.create_resource_utilization_chart(
                report_data, save_dir
            )
        )
        charts.append(resource_chart)

        # 3. Security score visualization
        security_chart = self.chart_generator.create_security_score_chart(
            report_data, save_dir
        )
        charts.append(security_chart)

        # 4. Compatibility matrix heatmap
        compatibility_chart = (
            self.chart_generator.create_compatibility_heatmap(
                report_data, save_dir
            )
        )
        charts.append(compatibility_chart)

        self.logger.info(f"Generated {len(charts)} performance charts")
        return charts
