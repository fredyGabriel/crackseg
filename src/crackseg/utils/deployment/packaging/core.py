"""Core packaging system for deployment artifacts.

This module provides the main PackagingSystem class with essential
packaging functionality, delegating specialized operations to submodules.
"""

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import ContainerizationConfig, PackagingResult

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
        self.supported_targets = ["docker", "kubernetes", "serverless", "edge"]
        self.base_images = {
            "python": "python:3.12-slim",
            "gpu": "nvidia/cuda:12.1-devel-ubuntu20.04",
            "minimal": "alpine:3.18",
            "ml": "python:3.12-slim",
        }

        # Containerization configurations
        self.container_configs = {
            "production": ContainerizationConfig(
                base_image="python:3.12-slim",
                non_root_user=True,
                security_scan=True,
                layer_caching=True,
                compression=True,
            ),
            "staging": ContainerizationConfig(
                base_image="python:3.12-slim",
                non_root_user=True,
                security_scan=False,
                layer_caching=True,
                compression=False,
            ),
            "development": ContainerizationConfig(
                base_image="python:3.12-slim",
                non_root_user=False,
                security_scan=False,
                layer_caching=False,
                compression=False,
            ),
        }

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
        package_dir = Path(f"deployments/{artifact_id}/package")
        package_dir.mkdir(parents=True, exist_ok=True)

        # Create standard directory structure
        directories = [
            "app",
            "config",
            "scripts",
            "tests",
            "docs",
            "k8s",
            "helm",
        ]
        for dir_name in directories:
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

    def _create_health_check(self, package_dir: Path) -> None:
        """Create health check script.

        Args:
            package_dir: Package directory
        """
        health_check_content = '''#!/usr/bin/env python3
"""Health check script for deployment."""

import requests
import sys
import time

def check_health(url: str, timeout: int = 30) -> bool:
    """Check service health."""
    try:
        response = requests.get(f"{url}/healthz", timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False

if __name__ == "__main__":
    service_url = "http://localhost:8501"
    max_attempts = 10

    for attempt in range(max_attempts):
        if check_health(service_url):
            print("✅ Service is healthy")
            sys.exit(0)
        print(f"⏳ Attempt {attempt + 1}/{max_attempts} - Service not ready")
        time.sleep(5)

    print("❌ Service health check failed")
    sys.exit(1)
'''
        health_script = package_dir / "scripts" / "health_check.py"
        health_script.write_text(health_check_content)
        health_script.chmod(0o755)

    def _create_config_files(
        self, package_dir: Path, config: "DeploymentConfig"
    ) -> None:
        """Create configuration files.

        Args:
            package_dir: Package directory
            config: Deployment configuration
        """
        # App configuration
        app_config = {
            "service_name": "crackseg",
            "version": "1.0.0",
            "environment": config.target_environment,
            "port": 8501,
            "host": "0.0.0.0",
        }

        import json

        config_file = package_dir / "config" / "app_config.json"
        config_file.write_text(json.dumps(app_config, indent=2))

        # Environment configuration
        env_config = {
            "environment": config.target_environment,
            "deployment_type": config.deployment_type,
            "target_format": config.target_format,
            "enable_quantization": config.enable_quantization,
            "enable_pruning": config.enable_pruning,
        }

        env_file = package_dir / "config" / "environment.json"
        env_file.write_text(json.dumps(env_config, indent=2))

    def _create_deployment_scripts(
        self, package_dir: Path, config: "DeploymentConfig"
    ) -> None:
        """Create deployment scripts.

        Args:
            package_dir: Package directory
            config: Deployment configuration
        """
        # Docker deployment script
        docker_script = """#!/bin/bash
set -e

echo "🐳 Building Docker image..."
docker build -t crackseg:latest .

echo "🚀 Starting container..."
docker run -d -p 8501:8501 --name crackseg-app crackseg:latest

echo "✅ Deployment completed"
"""
        docker_script_path = package_dir / "scripts" / "deploy_docker.sh"
        docker_script_path.write_text(docker_script)
        docker_script_path.chmod(0o755)

        # Kubernetes deployment script
        k8s_script = """#!/bin/bash
set -e

echo "☸️ Deploying to Kubernetes..."
kubectl apply -f k8s/

echo "⏳ Waiting for deployment..."
kubectl wait --for=condition=available --timeout=300s deployment/crackseg

echo "✅ Kubernetes deployment completed"
"""
        k8s_script_path = package_dir / "scripts" / "deploy_kubernetes.sh"
        k8s_script_path.write_text(k8s_script)
        k8s_script_path.chmod(0o755)

    def _generate_requirements(
        self, package_dir: Path, config: "DeploymentConfig"
    ) -> Path:
        """Generate requirements.txt with comprehensive dependencies.

        Args:
            package_dir: Package directory
            config: Deployment configuration

        Returns:
            Path to requirements.txt file
        """
        requirements_path = package_dir / "requirements.txt"

        # Core ML dependencies
        core_deps = [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "numpy>=1.21.0",
            "Pillow>=9.0.0",
            "opencv-python>=4.5.0",
        ]

        # Web framework dependencies
        web_deps = [
            "fastapi>=0.100.0",
            "uvicorn>=0.20.0",
            "streamlit>=1.25.0",
            "python-multipart>=0.0.6",
        ]

        # Monitoring and health check dependencies
        monitoring_deps = [
            "psutil>=5.9.0",
            "prometheus-client>=0.17.0",
            "requests>=2.28.0",
        ]

        # Security dependencies
        security_deps = [
            "bandit>=1.7.0",
            "safety>=2.3.0",
        ]

        # Development dependencies (for production builds)
        dev_deps = [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
        ]

        # Combine dependencies based on environment
        all_deps = core_deps + web_deps + monitoring_deps

        if config.target_environment == "production":
            all_deps.extend(security_deps)

        if config.target_environment in ["development", "staging"]:
            all_deps.extend(dev_deps)

        # Write requirements file
        requirements_content = "\n".join(all_deps) + "\n"
        requirements_path.write_text(requirements_content)

        return requirements_path

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

    def _calculate_package_metrics(
        self, package_dir: Path
    ) -> dict[str, float]:
        """Calculate package metrics.

        Args:
            package_dir: Package directory

        Returns:
            Dictionary with package metrics
        """
        total_size = 0
        file_count = 0

        for file_path in package_dir.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                file_count += 1

        # Count dependencies from requirements.txt
        requirements_file = package_dir / "requirements.txt"
        dependencies_count = 0
        if requirements_file.exists():
            dependencies_count = len(
                [
                    line
                    for line in requirements_file.read_text().splitlines()
                    if line.strip()
                ]
            )

        return {
            "size_mb": total_size / (1024 * 1024),
            "file_count": file_count,
            "dependencies_count": dependencies_count,
        }

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
