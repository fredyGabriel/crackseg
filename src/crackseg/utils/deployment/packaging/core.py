"""Core packaging system for deployment artifacts.

This module provides the main PackagingSystem class with essential
packaging functionality, delegating specialized operations to submodules.
"""

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import PackagingResult
from .constants import (
    BASE_IMAGES,
    PACKAGE_DIRECTORIES,
    SUPPORTED_TARGETS,
    default_container_configs,
)

if TYPE_CHECKING:
    from ..config import DeploymentConfig

logger = logging.getLogger(__name__)


class PackagingSystem:
    """Automated packaging system for deployment artifacts.

    Handles containerization, dependency management, and environment
    isolation for different deployment targets.
    """

    def __init__(self) -> None:
        """Initialize packaging system."""
        self.supported_targets = SUPPORTED_TARGETS
        self.base_images = BASE_IMAGES

        # Containerization configurations
        self.container_configs = default_container_configs()

        logger.info(
            "PackagingSystem initialized with advanced containerization"
        )

    def package_artifact(
        self, optimization_result: dict[str, Any], config: "DeploymentConfig"
    ) -> PackagingResult:
        """Package optimized artifact for deployment.

        Args:
            optimization_result: Result from artifact optimization
            config: Deployment configuration

        Returns:
            PackagingResult with detailed packaging information
        """
        logger.info(
            f"Packaging artifact for {config.deployment_type} deployment"
        )

        start_time = time.time()

        try:
            # Import specialized modules to avoid circular imports
            from .containerization import ContainerizationManager
            from .manifests import ManifestGenerator
            from .security import SecurityScanner

            # Initialize specialized managers
            container_manager = ContainerizationManager(self)
            manifest_generator = ManifestGenerator(self)
            security_scanner = SecurityScanner(self)

            # 1. Create deployment directory structure
            package_dir = self._create_package_structure(
                optimization_result, config
            )

            # 2. Generate requirements.txt with dependency analysis
            from .dependencies import DependencyManager

            dependency_manager = DependencyManager()
            requirements_path = dependency_manager.generate_requirements(
                package_dir, config
            )

            # 3. Create optimized Dockerfile
            dockerfile_path = container_manager.create_advanced_dockerfile(
                package_dir, requirements_path, config
            )

            # 4. Create comprehensive deployment manifests
            manifests = manifest_generator.create_deployment_manifests(
                package_dir, config
            )

            # 5. Build and optimize container image
            container_info = container_manager.build_optimized_container_image(
                package_dir, dockerfile_path, config
            )

            # 6. Perform security scanning if enabled
            security_results = {}
            if self._should_perform_security_scan(config):
                security_results = security_scanner.perform_security_scan(
                    container_info.get("image_name", "")
                )

            # 7. Calculate package metrics
            from .metrics import MetricsCalculator

            metrics_calculator = MetricsCalculator()
            package_metrics = metrics_calculator.calculate_package_metrics(
                package_dir
            )

            # 8. Push to registry if configured
            if config.target_environment == "production":
                container_manager.push_to_registry(
                    container_info.get("image_name", ""), config
                )

            build_time = time.time() - start_time

            return PackagingResult(
                success=True,
                package_path=str(package_dir),
                package_size_mb=package_metrics.get("size_mb", 0.0),
                container_image_name=container_info.get("image_name"),
                dockerfile_path=str(dockerfile_path),
                requirements_path=str(requirements_path),
                dependencies_count=int(
                    package_metrics.get("dependencies_count", 0)
                ),
                image_size_mb=container_info.get("image_size_mb", 0.0),
                build_time_seconds=build_time,
                layer_count=container_info.get("layer_count", 0),
                security_scan_results=security_results,
                kubernetes_manifests=manifests.get("kubernetes", []),
                docker_compose_path=manifests.get("docker_compose"),
                helm_chart_path=manifests.get("helm_chart"),
            )

        except Exception as e:
            logger.error(f"Packaging failed: {e}")
            return PackagingResult(success=False, error_message=str(e))

    def _create_package_structure(
        self, optimization_result: dict[str, Any], config: "DeploymentConfig"
    ) -> Path:
        """Create deployment package directory structure.

        Args:
            optimization_result: Optimization result data
            config: Deployment configuration

        Returns:
            Path to created package directory
        """
        artifact_id = optimization_result.get("artifact_id", "unknown")
        package_dir = Path(
            f"infrastructure/deployment/packages/{artifact_id}/package"
        )
        package_dir.mkdir(parents=True, exist_ok=True)

        # Create standard directory structure
        for dir_name in PACKAGE_DIRECTORIES:
            (package_dir / dir_name).mkdir(exist_ok=True)

        # Create app entrypoint
        # Use file generator for file creation
        from .file_generators import FileGenerator

        file_generator = FileGenerator()

        # Create application files
        file_generator.create_app_entrypoint(package_dir, config)
        file_generator.create_health_check(package_dir)
        file_generator.create_config_files(package_dir, config)
        file_generator.create_deployment_scripts(package_dir, config)

        return package_dir

    def _should_perform_security_scan(
        self, config: "DeploymentConfig"
    ) -> bool:
        """Check if security scan should be performed.

        Args:
            config: Deployment configuration

        Returns:
            True if security scan should be performed
        """
        return (
            config.target_environment == "production"
            and config.run_security_scan
        )

    def get_packaging_recommendations(
        self, config: "DeploymentConfig"
    ) -> dict[str, Any]:
        """Get packaging recommendations for configuration.

        Args:
            config: Deployment configuration

        Returns:
            Dictionary with packaging recommendations
        """
        recommendations: dict[str, Any] = {
            "target_environment": config.target_environment,
            "deployment_type": config.deployment_type,
            "recommended_base_image": self.base_images["python"],
        }

        # Optimization suggestions
        optimization_suggestions = []
        if config.target_environment == "production":
            optimization_suggestions.extend(
                [
                    "Use multi-stage Docker builds",
                    "Enable layer caching",
                    "Implement security scanning",
                    "Use non-root user",
                    "Enable image compression",
                ]
            )
        elif config.target_environment == "staging":
            optimization_suggestions.extend(
                [
                    "Use multi-stage builds for faster iteration",
                    "Enable layer caching for speed",
                    "Skip security scanning for speed",
                ]
            )
        else:  # development
            optimization_suggestions.extend(
                [
                    "Use single-stage builds for speed",
                    "Disable layer caching for faster iteration",
                    "Skip security scanning for speed",
                ]
            )

        recommendations["optimization_suggestions"] = optimization_suggestions

        # Security recommendations
        security_recommendations = []
        if config.target_environment == "production":
            security_recommendations.extend(
                [
                    "Regular vulnerability scans",
                    "Keep base images updated",
                    "Minimize attack surface",
                    "Use security-focused base images",
                ]
            )

        recommendations["security_recommendations"] = security_recommendations

        return recommendations

    def create_multi_target_package(
        self,
        optimization_result: dict[str, Any],
        configs: list["DeploymentConfig"],
    ) -> dict[str, PackagingResult]:
        """Create packages for multiple deployment targets.

        Args:
            optimization_result: Optimization result data
            configs: List of deployment configurations

        Returns:
            Dictionary mapping environment names to packaging results
        """
        results = {}

        for config in configs:
            try:
                result = self.package_artifact(optimization_result, config)
                results[config.target_environment] = result
            except Exception as e:
                logger.error(
                    f"Failed to package for {config.target_environment}: {e}"
                )
                results[config.target_environment] = PackagingResult(
                    success=False, error_message=str(e)
                )

        return results
