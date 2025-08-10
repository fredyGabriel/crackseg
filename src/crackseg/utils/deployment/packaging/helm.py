"""Helm chart generation for packaging system.

This module handles creation of Helm charts for Kubernetes deployments.
Heavy template content is provided by `helm_templates` to keep this file lean.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .helm_templates import (
    generate_chart_yaml,
    generate_configmap_template,
    generate_deployment_template,
    generate_hpa_template,
    generate_ingress_template,
    generate_notes_txt,
    generate_secret_template,
    generate_service_template,
    generate_values_yaml,
)

if TYPE_CHECKING:
    from ..config import DeploymentConfig

logger = logging.getLogger(__name__)


class HelmChartGenerator:
    """Generates Helm charts for Kubernetes deployments."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def create_helm_chart(
        self, package_dir: Path, config: DeploymentConfig
    ) -> str:
        """Create Helm chart for deployment.

        Args:
            package_dir: Package directory
            config: Deployment configuration

        Returns:
            Path string to created Helm chart directory
        """
        helm_dir = package_dir / "helm" / "crackseg"
        helm_dir.mkdir(parents=True, exist_ok=True)

        # Chart.yaml
        (helm_dir / "Chart.yaml").write_text(generate_chart_yaml())

        # values.yaml
        values_yaml = generate_values_yaml(
            replica_count=self._get_replica_count(config),
            is_production=(config.target_environment == "production"),
            env=config.target_environment,
            deployment_type=config.deployment_type,
            enable_health_checks=config.enable_health_checks,
            enable_metrics_collection=config.enable_metrics_collection,
        )
        (helm_dir / "values.yaml").write_text(values_yaml)

        # templates/
        templates_dir = helm_dir / "templates"
        templates_dir.mkdir(exist_ok=True)
        (templates_dir / "deployment.yaml").write_text(
            generate_deployment_template()
        )
        (templates_dir / "service.yaml").write_text(
            generate_service_template()
        )
        (templates_dir / "configmap.yaml").write_text(
            generate_configmap_template()
        )
        (templates_dir / "NOTES.txt").write_text(generate_notes_txt())

        if config.target_environment == "production":
            (templates_dir / "ingress.yaml").write_text(
                generate_ingress_template()
            )
            (templates_dir / "hpa.yaml").write_text(generate_hpa_template())
            (templates_dir / "secret.yaml").write_text(
                generate_secret_template()
            )

        self.logger.info(f"Created Helm chart in: {helm_dir}")
        return str(helm_dir)

    def _get_replica_count(self, config: DeploymentConfig) -> int:
        if config.target_environment == "production":
            return 3
        if config.target_environment == "staging":
            return 2
        return 1
