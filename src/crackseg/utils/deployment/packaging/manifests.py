"""Deployment manifests generation for packaging system.

This module handles creation of Kubernetes manifests, Helm charts,
and Docker Compose configurations by delegating to specialized modules.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..config import DeploymentConfig

logger = logging.getLogger(__name__)


class ManifestGenerator:
    """Generates deployment manifests for different targets."""

    def __init__(self, packaging_system: Any) -> None:
        """Initialize manifest generator.

        Args:
            packaging_system: Reference to main packaging system
        """
        self.packaging_system = packaging_system
        self.logger = logging.getLogger(__name__)

    def create_deployment_manifests(
        self, package_dir: Path, config: "DeploymentConfig"
    ) -> dict[str, Any]:
        """Create comprehensive deployment manifests.

        Args:
            package_dir: Package directory
            config: Deployment configuration

        Returns:
            Dictionary with manifest file paths
        """
        manifests = {}

        # Create Kubernetes manifests
        if config.deployment_type in ["kubernetes", "container"]:
            from .kubernetes import KubernetesManifestGenerator

            k8s_generator = KubernetesManifestGenerator()
            k8s_manifests = k8s_generator.create_kubernetes_manifests(
                package_dir, config
            )
            manifests["kubernetes"] = k8s_manifests

        # Create Helm chart
        if config.deployment_type == "kubernetes":
            from .helm import HelmChartGenerator

            helm_generator = HelmChartGenerator()
            helm_chart = helm_generator.create_helm_chart(package_dir, config)
            manifests["helm_chart"] = helm_chart

        # Create Docker Compose
        if config.deployment_type == "container":
            from .docker_compose import DockerComposeGenerator

            compose_generator = DockerComposeGenerator()
            docker_compose = compose_generator.create_docker_compose(
                package_dir, config
            )
            compose_generator.create_monitoring_configs(package_dir, config)
            manifests["docker_compose"] = docker_compose

        self.logger.info(
            f"Created manifests for {config.deployment_type} deployment"
        )
        return manifests
