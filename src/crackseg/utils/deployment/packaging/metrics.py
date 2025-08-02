"""Package metrics calculation for packaging system.

This module handles calculation of package metrics and recommendations.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..config import DeploymentConfig

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculates package metrics and recommendations."""

    def __init__(self) -> None:
        """Initialize metrics calculator."""
        self.logger = logging.getLogger(__name__)

    def calculate_package_metrics(self, package_dir: Path) -> dict[str, float]:
        """Calculate comprehensive package metrics.

        Args:
            package_dir: Package directory

        Returns:
            Dictionary with calculated metrics
        """
        metrics = {}

        try:
            # File count metrics
            metrics["total_files"] = self._count_files(package_dir)
            metrics["python_files"] = self._count_python_files(package_dir)
            metrics["config_files"] = self._count_config_files(package_dir)

            # Size metrics
            metrics["total_size_mb"] = self._calculate_total_size(package_dir)
            metrics["average_file_size_kb"] = (
                self._calculate_average_file_size(package_dir)
            )

            # Dependency metrics
            requirements_path = package_dir / "requirements.txt"
            if requirements_path.exists():
                metrics["dependencies_count"] = self._count_dependencies(
                    requirements_path
                )
            else:
                metrics["dependencies_count"] = 0

            # Complexity metrics
            metrics["complexity_score"] = self._calculate_complexity_score(
                package_dir
            )

            # Security metrics
            metrics["security_score"] = self._calculate_security_score(
                package_dir
            )

            self.logger.info(f"Calculated {len(metrics)} package metrics")

        except Exception as e:
            self.logger.error(f"Metrics calculation failed: {e}")
            metrics["error"] = str(e)

        return metrics

    def get_packaging_recommendations(
        self, config: "DeploymentConfig"
    ) -> dict[str, Any]:
        """Get packaging recommendations based on configuration.

        Args:
            config: Deployment configuration

        Returns:
            Dictionary with recommendations
        """
        recommendations = {
            "optimization": [],
            "security": [],
            "monitoring": [],
            "deployment": [],
        }

        # Optimization recommendations
        if config.deployment_type == "container":
            recommendations["optimization"].extend(
                [
                    "Use multi-stage Docker builds to reduce image size",
                    "Implement layer caching for faster builds",
                    "Consider using Alpine Linux for smaller base images",
                    "Optimize Python dependencies with pip-tools",
                ]
            )

        elif config.deployment_type == "kubernetes":
            recommendations["optimization"].extend(
                [
                    "Use resource limits and requests",
                    "Implement horizontal pod autoscaling",
                    "Consider using node affinity for GPU workloads",
                    "Use persistent volumes for model storage",
                ]
            )

        elif config.deployment_type == "serverless":
            recommendations["optimization"].extend(
                [
                    "Optimize cold start times with layer packaging",
                    "Use provisioned concurrency for consistent performance",
                    "Implement proper timeout configurations",
                    "Consider using container images for Lambda",
                ]
            )

        # Security recommendations
        if config.target_environment == "production":
            recommendations["security"].extend(
                [
                    "Run containers as non-root user",
                    "Implement security scanning with Trivy",
                    "Use secrets management for sensitive data",
                    "Enable network policies in Kubernetes",
                    "Implement proper RBAC configurations",
                ]
            )

        # Monitoring recommendations
        if config.enable_metrics_collection:
            recommendations["monitoring"].extend(
                [
                    "Implement Prometheus metrics collection",
                    "Set up health check endpoints",
                    "Configure log aggregation",
                    "Use distributed tracing for request tracking",
                    "Implement alerting for critical metrics",
                ]
            )

        # Deployment recommendations
        recommendations["deployment"].extend(
            [
                "Use blue-green deployment for zero-downtime updates",
                "Implement canary releases for gradual rollouts",
                "Set up automated rollback mechanisms",
                "Use infrastructure as code (Terraform/CloudFormation)",
                "Implement proper backup and disaster recovery",
            ]
        )

        return recommendations

    def _count_files(self, package_dir: Path) -> int:
        """Count total files in package directory."""
        count = 0
        for _ in package_dir.rglob("*"):
            if _.is_file():
                count += 1
        return count

    def _count_python_files(self, package_dir: Path) -> int:
        """Count Python files in package directory."""
        count = 0
        for file_path in package_dir.rglob("*.py"):
            if file_path.is_file():
                count += 1
        return count

    def _count_config_files(self, package_dir: Path) -> int:
        """Count configuration files in package directory."""
        config_extensions = {".yaml", ".yml", ".json", ".toml", ".ini", ".cfg"}
        count = 0
        for file_path in package_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix in config_extensions:
                count += 1
        return count

    def _calculate_total_size(self, package_dir: Path) -> float:
        """Calculate total size in MB."""
        total_size = 0
        for file_path in package_dir.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return round(total_size / (1024 * 1024), 2)

    def _calculate_average_file_size(self, package_dir: Path) -> float:
        """Calculate average file size in KB."""
        total_size = 0
        file_count = 0
        for file_path in package_dir.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                file_count += 1

        if file_count == 0:
            return 0.0

        return round((total_size / file_count) / 1024, 2)

    def _count_dependencies(self, requirements_path: Path) -> int:
        """Count dependencies in requirements.txt."""
        try:
            content = requirements_path.read_text()
            lines = [
                line.strip()
                for line in content.split("\n")
                if line.strip() and not line.startswith("#")
            ]
            return len(lines)
        except Exception:
            return 0

    def _calculate_complexity_score(self, package_dir: Path) -> float:
        """Calculate complexity score based on various factors."""
        score = 0.0

        # Factor in number of Python files
        python_files = self._count_python_files(package_dir)
        score += python_files * 0.1

        # Factor in number of dependencies
        requirements_path = package_dir / "requirements.txt"
        if requirements_path.exists():
            deps_count = self._count_dependencies(requirements_path)
            score += deps_count * 0.05

        # Factor in directory depth
        max_depth = 0
        for file_path in package_dir.rglob("*"):
            if file_path.is_file():
                depth = len(file_path.relative_to(package_dir).parts)
                max_depth = max(max_depth, depth)
        score += max_depth * 0.2

        return round(score, 2)

    def _calculate_security_score(self, package_dir: Path) -> float:
        """Calculate security score based on security measures."""
        score = 100.0  # Start with perfect score

        # Check for non-root user in Dockerfile
        dockerfile_path = package_dir / "Dockerfile"
        if dockerfile_path.exists():
            content = dockerfile_path.read_text()
            if "USER root" in content and "USER" not in content.replace(
                "USER root", ""
            ):
                score -= 20  # No non-root user specified

        # Check for security scanning
        if not any(package_dir.rglob("*.security")):
            score -= 15  # No security scan results

        # Check for secrets in files
        for file_path in package_dir.rglob("*.py"):
            if file_path.is_file():
                content = file_path.read_text()
                if any(
                    secret in content.lower()
                    for secret in ["password", "secret", "key"]
                ):
                    score -= 10  # Potential secrets in code

        return max(0.0, round(score, 2))
