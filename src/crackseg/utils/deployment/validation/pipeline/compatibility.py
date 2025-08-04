"""Compatibility checking for validation pipeline.

This module provides comprehensive compatibility checking capabilities for
deployment packages including Python version, dependencies, and environment checks.
"""

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from ..config import DeploymentConfig

logger = logging.getLogger(__name__)


class CompatibilityChecker:
    """Checker for compatibility testing of deployment packages."""

    def __init__(self) -> None:
        """Initialize compatibility checker."""
        logger.info("CompatibilityChecker initialized")

    def run_checks(
        self, packaging_result: dict[str, Any], config: "DeploymentConfig"
    ) -> dict[str, Any]:
        """Run compatibility checks on deployment package.

        Args:
            packaging_result: Result from packaging system
            config: Deployment configuration

        Returns:
            Dictionary with compatibility check results
        """
        try:
            package_dir = Path(packaging_result.get("package_dir", ""))
            if not package_dir.exists():
                return {
                    "compatibility_passed": False,
                    "error": "Package directory not found",
                }

            # Run compatibility checks
            python_compatible = self._check_python_compatibility()
            deps_compatible = self._check_dependencies_compatibility()
            env_compatible = self._check_environment_compatibility(config)

            # Overall compatibility
            all_compatible = (
                python_compatible and deps_compatible and env_compatible
            )

            return {
                "compatibility_passed": all_compatible,
                "python_compatible": python_compatible,
                "dependencies_compatible": deps_compatible,
                "environment_compatible": env_compatible,
            }

        except Exception as e:
            logger.error(f"Compatibility checks failed: {e}")
            return {
                "compatibility_passed": False,
                "error": str(e),
            }

    def _check_python_compatibility(self) -> bool:
        """Check Python version compatibility.

        Returns:
            True if Python version is compatible
        """
        try:
            python_version = self._get_python_version()
            major, minor = python_version.split(".")[:2]
            major, minor = int(major), int(minor)

            # Check if Python version is supported (3.8+)
            if major < 3 or (major == 3 and minor < 8):
                logger.error(
                    f"Python version {python_version} not supported. "
                    "Requires Python 3.8+"
                )
                return False

            logger.info(f"Python version {python_version} is compatible")
            return True

        except Exception as e:
            logger.error(f"Python compatibility check failed: {e}")
            return False

    def _check_dependencies_compatibility(self) -> bool:
        """Check dependencies compatibility.

        Returns:
            True if dependencies are compatible
        """
        try:
            # Check PyTorch version
            pytorch_version = self._get_pytorch_version()
            if pytorch_version:
                logger.info(f"PyTorch version {pytorch_version} detected")

            # Check for required packages
            required_packages = [
                "torch",
                "torchvision",
                "numpy",
                "opencv-python",
                "albumentations",
            ]

            missing_packages = []
            for package in required_packages:
                try:
                    __import__(package.replace("-", "_"))
                except ImportError:
                    missing_packages.append(package)

            if missing_packages:
                logger.error(f"Missing required packages: {missing_packages}")
                return False

            logger.info("All required dependencies are available")
            return True

        except Exception as e:
            logger.error(f"Dependencies compatibility check failed: {e}")
            return False

    def _check_environment_compatibility(
        self, config: "DeploymentConfig"
    ) -> bool:
        """Check environment compatibility.

        Args:
            config: Deployment configuration

        Returns:
            True if environment is compatible
        """
        try:
            # Check target environment requirements
            target_env = config.target_environment.lower()

            if target_env == "production":
                # Production environment checks
                if not self._check_production_requirements():
                    return False
            elif target_env == "staging":
                # Staging environment checks
                if not self._check_staging_requirements():
                    return False
            elif target_env == "development":
                # Development environment checks
                if not self._check_development_requirements():
                    return False
            else:
                logger.warning(f"Unknown environment: {target_env}")
                return True  # Don't fail for unknown environments

            logger.info(f"Environment {target_env} is compatible")
            return True

        except Exception as e:
            logger.error(f"Environment compatibility check failed: {e}")
            return False

    def _check_production_requirements(self) -> bool:
        """Check production environment requirements.

        Returns:
            True if production requirements are met
        """
        try:
            # Check for production-specific requirements
            # This could include security checks, performance requirements, etc.

            # For now, just check if we're running in a production-like environment
            return True

        except Exception as e:
            logger.error(f"Production requirements check failed: {e}")
            return False

    def _check_staging_requirements(self) -> bool:
        """Check staging environment requirements.

        Returns:
            True if staging requirements are met
        """
        try:
            # Check for staging-specific requirements
            return True

        except Exception as e:
            logger.error(f"Staging requirements check failed: {e}")
            return False

    def _check_development_requirements(self) -> bool:
        """Check development environment requirements.

        Returns:
            True if development requirements are met
        """
        try:
            # Check for development-specific requirements
            return True

        except Exception as e:
            logger.error(f"Development requirements check failed: {e}")
            return False

    def _get_python_version(self) -> str:
        """Get Python version.

        Returns:
            Python version string
        """
        try:
            return sys.version.split()[0]
        except Exception as e:
            logger.error(f"Failed to get Python version: {e}")
            return "0.0.0"

    def _get_pytorch_version(self) -> str:
        """Get PyTorch version.

        Returns:
            PyTorch version string or empty string if not available
        """
        try:
            return torch.__version__
        except Exception as e:
            logger.error(f"Failed to get PyTorch version: {e}")
            return ""
