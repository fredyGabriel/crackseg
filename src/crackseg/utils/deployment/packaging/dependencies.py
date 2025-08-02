"""Dependency management for packaging system.

This module handles requirements.txt generation and dependency analysis.
"""

import logging
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..config import DeploymentConfig

logger = logging.getLogger(__name__)


class DependencyManager:
    """Manages dependencies for packaging."""

    def __init__(self) -> None:
        """Initialize dependency manager."""
        self.logger = logging.getLogger(__name__)

    def generate_requirements(
        self, package_dir: Path, config: "DeploymentConfig"
    ) -> Path:
        """Generate requirements.txt with dependency analysis.

        Args:
            package_dir: Package directory
            config: Deployment configuration

        Returns:
            Path to generated requirements.txt
        """
        requirements_path = package_dir / "requirements.txt"

        # Core dependencies
        core_deps = self._get_core_dependencies()

        # Environment-specific dependencies
        env_deps = self._get_environment_dependencies(config)

        # Monitoring dependencies
        monitoring_deps = self._get_monitoring_dependencies(config)

        # Security dependencies
        security_deps = self._get_security_dependencies(config)

        # Development dependencies (for debugging)
        dev_deps = self._get_development_dependencies(config)

        # Combine all dependencies
        all_deps = (
            core_deps + env_deps + monitoring_deps + security_deps + dev_deps
        )

        # Write requirements.txt
        requirements_content = self._format_requirements(all_deps)
        requirements_path.write_text(requirements_content)

        self.logger.info(f"Generated requirements.txt: {requirements_path}")
        return requirements_path

    def _get_core_dependencies(self) -> list[str]:
        """Get core CrackSeg dependencies."""
        return [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "opencv-python>=4.8.0",
            "numpy>=1.24.0",
            "Pillow>=10.0.0",
            "scikit-image>=0.21.0",
            "matplotlib>=3.7.0",
            "hydra-core>=1.3.0",
            "omegaconf>=2.3.0",
            "pyyaml>=6.0",
        ]

    def _get_environment_dependencies(
        self, config: "DeploymentConfig"
    ) -> list[str]:
        """Get environment-specific dependencies."""
        deps = []

        if config.deployment_type == "serverless":
            deps.extend(
                [
                    "boto3>=1.28.0",
                    "aws-lambda-runtime-api>=0.2.0",
                ]
            )
        elif config.deployment_type == "kubernetes":
            deps.extend(
                [
                    "kubernetes>=28.0.0",
                    "kubernetes-asyncio>=28.0.0",
                ]
            )
        elif config.deployment_type == "container":
            deps.extend(
                [
                    "gunicorn>=21.0.0",
                    "uvicorn>=0.23.0",
                ]
            )

        return deps

    def _get_monitoring_dependencies(
        self, config: "DeploymentConfig"
    ) -> list[str]:
        """Get monitoring dependencies."""
        if not config.enable_metrics_collection:
            return []

        return [
            "prometheus-client>=0.17.0",
            "psutil>=5.9.0",
            "structlog>=23.0.0",
        ]

    def _get_security_dependencies(
        self, config: "DeploymentConfig"
    ) -> list[str]:
        """Get security dependencies."""
        return [
            "cryptography>=41.0.0",
            "safety>=2.3.0",
        ]

    def _get_development_dependencies(
        self, config: "DeploymentConfig"
    ) -> list[str]:
        """Get development dependencies."""
        if config.target_environment != "development":
            return []

        return [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "basedpyright>=1.8.0",
        ]

    def _format_requirements(self, dependencies: list[str]) -> str:
        """Format dependencies as requirements.txt content."""
        content = "# CrackSeg Dependencies\n"
        content += "# Generated automatically by packaging system\n\n"

        # Sort dependencies for consistency
        sorted_deps = sorted(dependencies)

        for dep in sorted_deps:
            content += f"{dep}\n"

        return content

    def analyze_dependencies(self, requirements_path: Path) -> dict[str, Any]:
        """Analyze dependencies for security and compatibility.

        Args:
            requirements_path: Path to requirements.txt

        Returns:
            Analysis results
        """
        analysis = {
            "total_dependencies": 0,
            "security_vulnerabilities": [],
            "outdated_packages": [],
            "compatibility_issues": [],
        }

        try:
            # Count dependencies
            requirements_content = requirements_path.read_text()
            lines = [
                line.strip()
                for line in requirements_content.split("\n")
                if line.strip() and not line.startswith("#")
            ]
            analysis["total_dependencies"] = len(lines)

            # Check for security vulnerabilities using safety
            safety_result = self._run_safety_check(requirements_path)
            analysis["security_vulnerabilities"] = safety_result

            self.logger.info(
                f"Dependency analysis completed: "
                f"{analysis['total_dependencies']} packages"
            )

        except Exception as e:
            self.logger.error(f"Dependency analysis failed: {e}")
            analysis["error"] = str(e)

        return analysis

    def _run_safety_check(self, requirements_path: Path) -> list[str]:
        """Run safety check for vulnerabilities."""
        try:
            result = subprocess.run(
                ["safety", "check", "-r", str(requirements_path)],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                return []
            else:
                # Parse safety output for vulnerabilities
                vulnerabilities = []
                for line in result.stdout.split("\n"):
                    if "Vulnerability" in line:
                        vulnerabilities.append(line.strip())
                return vulnerabilities

        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.logger.warning("Safety check not available")
            return []
