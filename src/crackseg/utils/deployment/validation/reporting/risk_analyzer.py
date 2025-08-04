"""Risk analysis for validation reporting."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..config import DeploymentConfig


class RiskAnalyzer:
    """Analyzer for deployment risk assessment."""

    def calculate_risk_level(self, validation_results: dict[str, Any]) -> str:
        """Calculate deployment risk level.

        Args:
            validation_results: Results from validation pipeline

        Returns:
            Risk level: "low", "medium", "high", "critical"
        """
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

    def generate_recommendations(
        self, validation_results: dict[str, Any], config: "DeploymentConfig"
    ) -> list[str]:
        """Generate actionable recommendations.

        Args:
            validation_results: Results from validation pipeline
            config: Deployment configuration

        Returns:
            List of recommendations
        """
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

    def generate_warnings(
        self, validation_results: dict[str, Any]
    ) -> list[str]:
        """Generate warnings for potential issues.

        Args:
            validation_results: Results from validation pipeline

        Returns:
            List of warnings
        """
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

    def generate_critical_issues(
        self, validation_results: dict[str, Any]
    ) -> list[str]:
        """Generate critical issues that must be addressed.

        Args:
            validation_results: Results from validation pipeline

        Returns:
            List of critical issues
        """
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

    def estimate_deployment_time(
        self, validation_results: dict[str, Any], config: "DeploymentConfig"
    ) -> int:
        """Estimate deployment time in minutes.

        Args:
            validation_results: Results from validation pipeline
            config: Deployment configuration

        Returns:
            Estimated deployment time in minutes
        """
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
