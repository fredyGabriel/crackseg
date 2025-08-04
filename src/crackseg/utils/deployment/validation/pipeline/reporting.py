"""Report generation for validation pipeline.

This module provides comprehensive report generation capabilities for
validation results including detailed analysis and recommendations.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..config import DeploymentConfig

logger = logging.getLogger(__name__)


class ValidationReporter:
    """Reporter for validation results and analysis."""

    def __init__(self) -> None:
        """Initialize validation reporter."""
        logger.info("ValidationReporter initialized")

    def generate_report(
        self, validation_results: dict[str, Any], config: "DeploymentConfig"
    ) -> str:
        """Generate comprehensive validation report.

        Args:
            validation_results: Validation results dictionary
            config: Deployment configuration

        Returns:
            Formatted validation report
        """
        try:
            report_lines = []

            # Header
            report_lines.append("=" * 80)
            report_lines.append("CRACKSEG DEPLOYMENT VALIDATION REPORT")
            report_lines.append("=" * 80)
            report_lines.append(
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            report_lines.append(f"Artifact ID: {config.artifact_id}")
            report_lines.append(
                f"Target Environment: {config.target_environment}"
            )
            report_lines.append(f"Target Format: {config.target_format}")
            report_lines.append("")

            # Overall Status
            overall_success = validation_results.get("success", False)
            status_icon = "✅" if overall_success else "❌"
            status_text = "PASSED" if overall_success else "FAILED"

            report_lines.append(
                f"{status_icon} OVERALL VALIDATION: {status_text}"
            )
            report_lines.append("")

            # Functional Tests
            functional_passed = validation_results.get(
                "functional_tests_passed", False
            )
            func_icon = "✅" if functional_passed else "❌"
            report_lines.append(
                f"{func_icon} Functional Tests: {'PASSED' if functional_passed else 'FAILED'}"
            )

            if "test_output" in validation_results:
                report_lines.append("   Test Output:")
                for line in validation_results["test_output"].split("\n")[:10]:
                    if line.strip():
                        report_lines.append(f"     {line}")
                if len(validation_results["test_output"].split("\n")) > 10:
                    report_lines.append("     ... (truncated)")

            # Performance Metrics
            performance_score = validation_results.get(
                "performance_score", 0.0
            )
            inference_time = validation_results.get("inference_time_ms", 0.0)
            memory_usage = validation_results.get("memory_usage_mb", 0.0)
            throughput = validation_results.get("throughput_rps", 0.0)

            perf_icon = (
                "✅"
                if performance_score >= 0.7
                else "⚠️" if performance_score >= 0.5 else "❌"
            )
            report_lines.append("")
            report_lines.append(f"{perf_icon} Performance Metrics:")
            report_lines.append(f"   Score: {performance_score:.2f}/1.0")
            report_lines.append(f"   Inference Time: {inference_time:.2f}ms")
            report_lines.append(f"   Memory Usage: {memory_usage:.2f}MB")
            report_lines.append(f"   Throughput: {throughput:.2f} RPS")

            # Security Scan
            security_passed = validation_results.get(
                "security_scan_passed", False
            )
            vulnerabilities = validation_results.get(
                "vulnerabilities_found", 0
            )
            security_score = validation_results.get("security_score", 0.0)

            sec_icon = "✅" if security_passed else "❌"
            report_lines.append("")
            report_lines.append(f"{sec_icon} Security Scan:")
            report_lines.append(
                f"   Status: {'PASSED' if security_passed else 'FAILED'}"
            )
            report_lines.append(f"   Vulnerabilities Found: {vulnerabilities}")
            report_lines.append(
                f"   Security Score: {security_score:.1f}/10.0"
            )

            # Compatibility Checks
            compatibility_passed = validation_results.get(
                "compatibility_passed", True
            )
            compat_icon = "✅" if compatibility_passed else "❌"
            report_lines.append("")
            report_lines.append(f"{compat_icon} Compatibility Checks:")
            report_lines.append(
                f"   Status: {'PASSED' if compatibility_passed else 'FAILED'}"
            )

            # Detailed Results
            if "validation_details" in validation_results:
                report_lines.append("")
                report_lines.append("📊 DETAILED RESULTS:")
                report_lines.append("-" * 40)

                details = validation_results["validation_details"]
                for key, value in details.items():
                    if key not in ["success", "validation_details"]:
                        if isinstance(value, bool):
                            icon = "✅" if value else "❌"
                            report_lines.append(
                                f"{icon} {key}: {'PASSED' if value else 'FAILED'}"
                            )
                        elif isinstance(value, int | float):
                            report_lines.append(f"📈 {key}: {value}")
                        elif isinstance(value, str):
                            report_lines.append(f"📝 {key}: {value}")

            # Recommendations
            report_lines.append("")
            report_lines.append("💡 RECOMMENDATIONS:")
            report_lines.append("-" * 40)

            recommendations = self._generate_recommendations(
                validation_results
            )
            for rec in recommendations:
                report_lines.append(f"• {rec}")

            # Footer
            report_lines.append("")
            report_lines.append("=" * 80)
            report_lines.append("END OF VALIDATION REPORT")
            report_lines.append("=" * 80)

            return "\n".join(report_lines)

        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return f"Error generating report: {e}"

    def _generate_recommendations(
        self, validation_results: dict[str, Any]
    ) -> list[str]:
        """Generate recommendations based on validation results.

        Args:
            validation_results: Validation results dictionary

        Returns:
            List of recommendations
        """
        recommendations = []

        # Performance recommendations
        performance_score = validation_results.get("performance_score", 0.0)
        if performance_score < 0.7:
            recommendations.append(
                "Consider model optimization to improve performance"
            )

        inference_time = validation_results.get("inference_time_ms", 0.0)
        if inference_time > 1000:
            recommendations.append(
                "Inference time is high - consider model quantization or optimization"
            )

        memory_usage = validation_results.get("memory_usage_mb", 0.0)
        if memory_usage > 2048:
            recommendations.append(
                "Memory usage is high - consider model compression or smaller architecture"
            )

        # Security recommendations
        vulnerabilities = validation_results.get("vulnerabilities_found", 0)
        if vulnerabilities > 0:
            recommendations.append(
                f"Address {vulnerabilities} security vulnerabilities before deployment"
            )

        security_score = validation_results.get("security_score", 0.0)
        if security_score < 8.0:
            recommendations.append(
                "Improve security posture - review dependencies and configurations"
            )

        # Functional test recommendations
        functional_passed = validation_results.get(
            "functional_tests_passed", False
        )
        if not functional_passed:
            recommendations.append(
                "Fix functional test failures before deployment"
            )

        # General recommendations
        if not recommendations:
            recommendations.append(
                "All validation checks passed - ready for deployment"
            )

        return recommendations

    def save_report(
        self, report: str, output_path: Path, config: "DeploymentConfig"
    ) -> bool:
        """Save validation report to file.

        Args:
            report: Validation report content
            output_path: Path to save report
            config: Deployment configuration

        Returns:
            True if report saved successfully
        """
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save report
            output_path.write_text(report, encoding="utf-8")

            logger.info(f"Validation report saved to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save validation report: {e}")
            return False

    def generate_json_report(
        self, validation_results: dict[str, Any], config: "DeploymentConfig"
    ) -> dict[str, Any]:
        """Generate JSON format validation report.

        Args:
            validation_results: Validation results dictionary
            config: Deployment configuration

        Returns:
            JSON format validation report
        """
        try:
            json_report = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "artifact_id": config.artifact_id,
                    "target_environment": config.target_environment,
                    "target_format": config.target_format,
                },
                "overall_status": {
                    "success": validation_results.get("success", False),
                    "passed_checks": [],
                    "failed_checks": [],
                },
                "functional_tests": {
                    "passed": validation_results.get(
                        "functional_tests_passed", False
                    ),
                    "output": validation_results.get("test_output", ""),
                    "errors": validation_results.get("test_errors", ""),
                },
                "performance_metrics": {
                    "score": validation_results.get("performance_score", 0.0),
                    "inference_time_ms": validation_results.get(
                        "inference_time_ms", 0.0
                    ),
                    "memory_usage_mb": validation_results.get(
                        "memory_usage_mb", 0.0
                    ),
                    "throughput_rps": validation_results.get(
                        "throughput_rps", 0.0
                    ),
                },
                "security_scan": {
                    "passed": validation_results.get(
                        "security_scan_passed", False
                    ),
                    "vulnerabilities_found": validation_results.get(
                        "vulnerabilities_found", 0
                    ),
                    "security_score": validation_results.get(
                        "security_score", 0.0
                    ),
                },
                "compatibility": {
                    "passed": validation_results.get(
                        "compatibility_passed", True
                    ),
                },
                "recommendations": self._generate_recommendations(
                    validation_results
                ),
            }

            return json_report

        except Exception as e:
            logger.error(f"JSON report generation failed: {e}")
            return {"error": str(e)}
