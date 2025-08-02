"""Comprehensive validation reporting system.

This module provides advanced reporting capabilities for deployment validation,
including performance metrics, resource utilization, compatibility matrices,
and actionable deployment recommendations.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from jinja2 import Environment, FileSystemLoader

from .config import DeploymentConfig

logger = logging.getLogger(__name__)


@dataclass
class ValidationReportData:
    """Data structure for comprehensive validation reports."""

    # Basic information
    artifact_id: str
    target_environment: str
    target_format: str
    validation_timestamp: str
    validation_duration: float

    # Performance metrics
    inference_time_ms: float
    memory_usage_mb: float
    throughput_rps: float
    performance_score: float

    # Security metrics
    security_score: float
    vulnerabilities_found: int
    security_scan_passed: bool

    # Compatibility metrics
    compatibility_score: float
    python_compatible: bool
    dependencies_compatible: bool
    environment_compatible: bool

    # Functional test results
    functional_tests_passed: bool
    test_coverage_percentage: float
    critical_tests_passed: bool

    # Resource utilization
    cpu_usage_percent: float
    gpu_usage_percent: float
    disk_usage_mb: float
    network_bandwidth_mbps: float

    # Deployment readiness
    deployment_ready: bool
    risk_level: str  # "low", "medium", "high", "critical"
    estimated_deployment_time: int  # minutes

    # Recommendations
    recommendations: list[str]
    warnings: list[str]
    critical_issues: list[str]


class ValidationReporter:
    """Advanced validation reporting system."""

    def __init__(self, output_dir: Path | None = None) -> None:
        """Initialize validation reporter.

        Args:
            output_dir: Directory to save reports.
            Defaults to outputs/validation_reports/
        """
        self.output_dir = output_dir or Path("outputs/validation_reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup Jinja2 environment for templating
        self.template_env = Environment(
            loader=FileSystemLoader(Path(__file__).parent / "templates"),
            autoescape=True,
        )

        logger.info(
            f"ValidationReporter initialized with output_dir: "
            f"{self.output_dir}"
        )

    def generate_comprehensive_report(
        self,
        validation_results: dict[str, Any],
        config: DeploymentConfig,
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
        logger.info(
            f"Generating comprehensive report for {config.artifact_id}"
        )

        # Calculate risk level
        risk_level = self._calculate_risk_level(validation_results)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            validation_results, config
        )
        warnings = self._generate_warnings(validation_results)
        critical_issues = self._generate_critical_issues(validation_results)

        # Estimate deployment time
        deployment_time = self._estimate_deployment_time(
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
            content = self._generate_markdown_report(report_data)
        elif format == "html":
            report_path = self.output_dir / f"{filename}.html"
            content = self._generate_html_report(report_data)
        elif format == "json":
            report_path = self.output_dir / f"{filename}.json"
            content = json.dumps(
                self._report_data_to_dict(report_data), indent=2
            )
        elif format == "pdf":
            report_path = self.output_dir / f"{filename}.pdf"
            content = self._generate_pdf_report(report_data)
        else:
            raise ValueError(f"Unsupported format: {format}")

        report_path.write_text(content)
        logger.info(f"Report saved to: {report_path}")

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
        radar_chart = self._create_performance_radar_chart(
            report_data, save_dir
        )
        charts.append(radar_chart)

        # 2. Resource utilization bar chart
        resource_chart = self._create_resource_utilization_chart(
            report_data, save_dir
        )
        charts.append(resource_chart)

        # 3. Security score visualization
        security_chart = self._create_security_score_chart(
            report_data, save_dir
        )
        charts.append(security_chart)

        # 4. Compatibility matrix heatmap
        compatibility_chart = self._create_compatibility_heatmap(
            report_data, save_dir
        )
        charts.append(compatibility_chart)

        logger.info(f"Generated {len(charts)} performance charts")
        return charts

    def _calculate_risk_level(self, validation_results: dict[str, Any]) -> str:
        """Calculate deployment risk level."""
        risk_score = 0

        # Performance risks
        if validation_results.get("performance_score", 0.0) < 0.7:
            risk_score += 2
        if validation_results.get("inference_time_ms", 0.0) > 1000:
            risk_score += 1

        # Security risks
        if not validation_results.get("security_scan_passed", False):
            risk_score += 3
        if validation_results.get("vulnerabilities_found", 0) > 0:
            risk_score += 2

        # Compatibility risks
        if not validation_results.get("functional_tests_passed", False):
            risk_score += 3
        if validation_results.get("compatibility_score", 0.0) < 0.8:
            risk_score += 1

        # Determine risk level
        if risk_score >= 6:
            return "critical"
        elif risk_score >= 4:
            return "high"
        elif risk_score >= 2:
            return "medium"
        else:
            return "low"

    def _generate_recommendations(
        self, validation_results: dict[str, Any], config: DeploymentConfig
    ) -> list[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Performance recommendations
        if validation_results.get("performance_score", 0.0) < 0.8:
            recommendations.append(
                "Consider model optimization (quantization, pruning) "
                "to improve performance"
            )

        if validation_results.get("inference_time_ms", 0.0) > 500:
            recommendations.append(
                "Optimize inference pipeline or consider hardware upgrade "
                "for faster inference"
            )

        # Security recommendations
        if not validation_results.get("security_scan_passed", False):
            recommendations.append(
                "Address security vulnerabilities before deployment"
            )

        if validation_results.get("vulnerabilities_found", 0) > 0:
            vuln_count = validation_results.get("vulnerabilities_found")
            recommendations.append(
                f"Review and fix {vuln_count} security vulnerabilities"
            )

        # Compatibility recommendations
        if not validation_results.get("functional_tests_passed", False):
            recommendations.append(
                "Fix functional test failures before deployment"
            )

        if validation_results.get("compatibility_score", 0.0) < 0.9:
            msg = (
                "Verify environment compatibility and update dependencies "
                "if needed"
            )
            recommendations.append(msg)

        # Environment-specific recommendations
        if config.target_environment == "kubernetes":
            msg = (
                "Ensure Kubernetes manifests are properly configured "
                "for resource limits"
            )
            recommendations.append(msg)
        elif config.target_environment == "docker":
            recommendations.append(
                "Optimize Docker image size and layer caching"
            )

        return recommendations

    def _generate_warnings(
        self, validation_results: dict[str, Any]
    ) -> list[str]:
        """Generate warnings for potential issues."""
        warnings = []

        # Performance warnings
        if validation_results.get("memory_usage_mb", 0.0) > 2048:
            memory_usage = validation_results.get("memory_usage_mb")
            warnings.append(f"High memory usage: {memory_usage:.1f}MB")

        if validation_results.get("throughput_requests_per_second", 0.0) < 10:
            throughput = validation_results.get(
                "throughput_requests_per_second"
            )
            warnings.append(f"Low throughput: {throughput:.1f} RPS")

        # Security warnings
        if validation_results.get("security_score", 0.0) < 9.0:
            security_score = validation_results.get("security_score")
            warnings.append(
                f"Security score below threshold: {security_score:.1f}/10.0"
            )

        return warnings

    def _generate_critical_issues(
        self, validation_results: dict[str, Any]
    ) -> list[str]:
        """Generate critical issues that must be addressed."""
        critical_issues = []

        # Critical functional issues
        if not validation_results.get("functional_tests_passed", False):
            critical_issues.append(
                "Functional tests failed - deployment blocked"
            )

        if not validation_results.get("security_scan_passed", False):
            critical_issues.append("Security scan failed - deployment blocked")

        # Critical performance issues
        if validation_results.get("performance_score", 0.0) < 0.5:
            critical_issues.append(
                "Performance score critically low - review required"
            )

        return critical_issues

    def _estimate_deployment_time(
        self, validation_results: dict[str, Any], config: DeploymentConfig
    ) -> int:
        """Estimate deployment time in minutes."""
        base_time = 5  # Base deployment time

        # Add time for performance issues
        if validation_results.get("performance_score", 0.0) < 0.7:
            base_time += 10

        # Add time for security issues
        if not validation_results.get("security_scan_passed", False):
            base_time += 15

        # Add time for compatibility issues
        if validation_results.get("compatibility_score", 0.0) < 0.8:
            base_time += 10

        # Environment-specific adjustments
        if config.target_environment == "kubernetes":
            base_time += 5
        elif config.target_environment == "docker":
            base_time += 3

        return base_time

    def _generate_markdown_report(
        self, report_data: ValidationReportData
    ) -> str:
        """Generate markdown format report."""
        template = self.template_env.get_template("validation_report.md.j2")
        return template.render(report=report_data)

    def _generate_html_report(self, report_data: ValidationReportData) -> str:
        """Generate HTML format report."""
        template = self.template_env.get_template("validation_report.html.j2")
        return template.render(report=report_data)

    def _generate_pdf_report(self, report_data: ValidationReportData) -> str:
        """Generate PDF format report."""
        # Convert markdown to PDF using pandoc or similar
        markdown_content = self._generate_markdown_report(report_data)
        # This would require additional PDF generation logic
        return markdown_content  # Placeholder

    def _report_data_to_dict(
        self, report_data: ValidationReportData
    ) -> dict[str, Any]:
        """Convert report data to dictionary for JSON export."""
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

    def _create_performance_radar_chart(
        self, report_data: ValidationReportData, save_dir: Path
    ) -> Path:
        """Create performance radar chart."""
        # Performance metrics for radar chart
        categories = [
            "Performance Score",
            "Security Score",
            "Compatibility Score",
            "Functional Tests",
            "Resource Efficiency",
        ]

        values = [
            report_data.performance_score * 100,
            report_data.security_score * 10,
            report_data.compatibility_score * 100,
            100 if report_data.functional_tests_passed else 0,
            max(0, 100 - (report_data.memory_usage_mb / 2048) * 100),
        ]

        # Create radar chart
        angles = [
            i / len(categories) * 2 * 3.14159 for i in range(len(categories))
        ]
        angles += angles[:1]  # Close the loop
        values += values[:1]

        _, ax = plt.subplots(
            figsize=(8, 8), subplot_kw={"projection": "polar"}
        )
        ax.plot(angles, values, "o-", linewidth=2, label="Validation Scores")
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 100)
        ax.set_title("Validation Performance Radar Chart", size=16, pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

        chart_path = save_dir / "performance_radar.png"
        plt.savefig(chart_path, dpi=300, bbox_inches="tight")
        plt.close()

        return chart_path

    def _create_resource_utilization_chart(
        self, report_data: ValidationReportData, save_dir: Path
    ) -> Path:
        """Create resource utilization bar chart."""
        resources = ["CPU", "GPU", "Memory (GB)", "Disk (MB)"]
        values = [
            report_data.cpu_usage_percent,
            report_data.gpu_usage_percent,
            report_data.memory_usage_mb / 1024,  # Convert to GB
            report_data.disk_usage_mb,
        ]

        _, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(
            resources,
            values,
            color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"],
        )

        # Add value labels on bars
        for bar, value in zip(bars, values, strict=False):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{value:.1f}",
                ha="center",
                va="bottom",
            )

        ax.set_title("Resource Utilization", fontsize=14, pad=20)
        ax.set_ylabel("Usage (%)")
        ax.set_ylim(0, max(values) * 1.1)

        chart_path = save_dir / "resource_utilization.png"
        plt.savefig(chart_path, dpi=300, bbox_inches="tight")
        plt.close()

        return chart_path

    def _create_security_score_chart(
        self, report_data: ValidationReportData, save_dir: Path
    ) -> Path:
        """Create security score visualization."""
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Security score gauge
        score = report_data.security_score
        ax1.pie(
            [score, 10 - score],
            colors=["#FF6B6B" if score < 8 else "#4ECDC4", "#F0F0F0"],
            startangle=90,
            counterclock=False,
        )
        ax1.text(
            0,
            0,
            f"{score:.1f}/10",
            ha="center",
            va="center",
            fontsize=20,
            fontweight="bold",
        )
        ax1.set_title("Security Score", fontsize=14)

        # Vulnerabilities bar chart
        vulnerabilities = report_data.vulnerabilities_found
        ax2.bar(
            ["Vulnerabilities"],
            [vulnerabilities],
            color="#FF6B6B" if vulnerabilities > 0 else "#4ECDC4",
        )
        ax2.set_title("Security Vulnerabilities Found", fontsize=14)
        ax2.set_ylabel("Count")
        ax2.text(
            0,
            vulnerabilities + 0.1,
            str(vulnerabilities),
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="bold",
        )

        plt.tight_layout()
        chart_path = save_dir / "security_score.png"
        plt.savefig(chart_path, dpi=300, bbox_inches="tight")
        plt.close()

        return chart_path

    def _create_compatibility_heatmap(
        self, report_data: ValidationReportData, save_dir: Path
    ) -> Path:
        """Create compatibility matrix heatmap."""
        compatibility_matrix = [
            ["Python", "Dependencies", "Environment"],
            [
                report_data.python_compatible,
                report_data.dependencies_compatible,
                report_data.environment_compatible,
            ],
        ]

        df = pd.DataFrame(
            compatibility_matrix[1:], columns=compatibility_matrix[0]
        )
        df_numeric = df.astype(int)

        _fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(
            df_numeric,
            annot=df,
            fmt="s",
            cmap="RdYlGn",
            cbar_kws={"label": "Compatibility Status"},
        )
        ax.set_title("Compatibility Matrix", fontsize=14, pad=20)
        ax.set_ylabel("Components")

        chart_path = save_dir / "compatibility_heatmap.png"
        plt.savefig(chart_path, dpi=300, bbox_inches="tight")
        plt.close()

        return chart_path
