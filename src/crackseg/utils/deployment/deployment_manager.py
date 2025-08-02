"""Deployment manager for CrackSeg.

This module provides the main deployment orchestration capabilities,
integrating artifact selection, environment configuration, optimization,
packaging, and deployment orchestration.
"""

import logging
from pathlib import Path
from typing import Any

from ..traceability import TraceabilityStorage
from .artifact_optimizer import ArtifactOptimizer
from .artifact_selector import ArtifactSelector, SelectionCriteria
from .config import DeploymentConfig, DeploymentResult
from .environment_configurator import EnvironmentConfigurator
from .monitoring_system import DeploymentMonitoringSystem
from .orchestration import DeploymentOrchestrator, DeploymentStrategy
from .packaging_system import PackagingSystem
from .validation_pipeline import ValidationPipeline


class DeploymentManager:
    """Main deployment manager for CrackSeg.

    Orchestrates the entire deployment pipeline including artifact selection,
    environment configuration, optimization, packaging, and deployment.
    """

    def __init__(
        self, storage: TraceabilityStorage, output_dir: Path | None = None
    ) -> None:
        """Initialize deployment manager.

        Args:
            storage: Traceability storage for artifact management
            output_dir: Output directory for deployment artifacts
        """
        self.storage = storage
        self.output_dir = output_dir or Path("deployments")
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.artifact_selector = ArtifactSelector(storage.query_interface)
        self.environment_configurator = EnvironmentConfigurator()
        self.artifact_optimizer = ArtifactOptimizer()
        self.packaging_system = PackagingSystem()
        self.validation_pipeline = ValidationPipeline()
        self.monitoring_system = DeploymentMonitoringSystem()
        self.orchestrator = DeploymentOrchestrator()

        self.logger.info("DeploymentManager initialized")

    def deploy_artifact(
        self,
        artifact_id: str,
        target_environment: str = "production",
        deployment_type: str = "container",
        enable_quantization: bool = True,
        target_format: str = "onnx",
        strategy: DeploymentStrategy = DeploymentStrategy.BLUE_GREEN,
        **kwargs,
    ) -> DeploymentResult:
        """Deploy artifact through complete pipeline with orchestration.

        Args:
            artifact_id: ID of artifact to deploy
            target_environment: Target deployment environment
            deployment_type: Type of deployment
            enable_quantization: Enable model quantization
            target_format: Target model format
            strategy: Deployment strategy to use
            **kwargs: Additional deployment options

        Returns:
            Deployment result
        """
        self.logger.info(
            f"Starting {strategy.value} deployment of artifact {artifact_id} "
            f"to {target_environment}"
        )

        try:
            # 1. Create deployment configuration
            config = DeploymentConfig(
                artifact_id=artifact_id,
                target_environment=target_environment,
                deployment_type=deployment_type,
                enable_quantization=enable_quantization,
                target_format=target_format,
                **kwargs,
            )

            # 2. Select appropriate artifact
            selection_result = self._select_artifact(config)
            if not selection_result.success:
                return DeploymentResult(
                    success=False,
                    deployment_id=f"failed-{artifact_id}",
                    artifact_id=artifact_id,
                    target_environment=target_environment,
                    error_message=selection_result.error_message,
                )

            # 3. Configure environment
            env_config = self.environment_configurator.configure_environment(
                config
            )

            # 4. Optimize artifact
            optimization_result = self.artifact_optimizer.optimize_artifact(
                selection_result.artifact, config
            )

            # 5. Package artifact
            packaging_result = self.packaging_system.package_artifact(
                optimization_result, config
            )

            # 6. Validate deployment
            validation_result = self.validation_pipeline.validate_deployment(
                packaging_result, config
            )

            if not validation_result.success:
                return DeploymentResult(
                    success=False,
                    deployment_id=f"validation-failed-{artifact_id}",
                    artifact_id=artifact_id,
                    target_environment=target_environment,
                    error_message=validation_result.error_message,
                )

            # 7. Deploy with orchestration strategy
            deployment_result = self.orchestrator.deploy_with_strategy(
                config,
                strategy,
                self._execute_deployment,
                env_config=env_config,
                packaging_result=packaging_result,
            )

            # 8. Start monitoring
            if deployment_result.success:
                self.monitoring_system.start_monitoring(
                    deployment_result.deployment_id,
                    deployment_result.deployment_url,
                    config,
                )

            return deployment_result

        except Exception as e:
            self.logger.error(f"Deployment failed: {e}")
            return DeploymentResult(
                success=False,
                deployment_id=f"error-{artifact_id}",
                artifact_id=artifact_id,
                target_environment=target_environment,
                error_message=str(e),
            )

    def _execute_deployment(
        self,
        config: DeploymentConfig,
        env_config: Any,
        packaging_result: Any,
        **kwargs,
    ) -> DeploymentResult:
        """Execute actual deployment to target environment.

        Args:
            config: Deployment configuration
            env_config: Environment configuration
            packaging_result: Packaging result
            **kwargs: Additional deployment parameters

        Returns:
            Deployment result
        """
        deployment_id = f"{config.artifact_id}-{config.target_environment}"

        try:
            if config.deployment_type == "container":
                result = self._deploy_container(
                    config, env_config, packaging_result
                )
            elif config.deployment_type == "kubernetes":
                result = self._deploy_kubernetes(
                    config, env_config, packaging_result
                )
            elif config.deployment_type == "serverless":
                result = self._deploy_serverless(
                    config, env_config, packaging_result
                )
            elif config.deployment_type == "edge":
                result = self._deploy_edge(
                    config, env_config, packaging_result
                )
            else:
                raise ValueError(
                    f"Unsupported deployment type: {config.deployment_type}"
                )

            return DeploymentResult(
                success=True,
                deployment_id=deployment_id,
                artifact_id=config.artifact_id,
                target_environment=config.target_environment,
                deployment_url=result.get("deployment_url"),
                health_check_url=result.get("health_check_url"),
                monitoring_dashboard_url=result.get(
                    "monitoring_dashboard_url"
                ),
            )

        except Exception as e:
            self.logger.error(f"Deployment execution failed: {e}")
            return DeploymentResult(
                success=False,
                deployment_id=deployment_id,
                artifact_id=config.artifact_id,
                target_environment=config.target_environment,
                error_message=str(e),
            )

    def deploy_with_rollback(
        self,
        artifact_id: str,
        target_environment: str = "production",
        strategy: DeploymentStrategy = DeploymentStrategy.BLUE_GREEN,
        **kwargs,
    ) -> DeploymentResult:
        """Deploy with automatic rollback on failure.

        Args:
            artifact_id: ID of artifact to deploy
            target_environment: Target deployment environment
            strategy: Deployment strategy to use
            **kwargs: Additional deployment options

        Returns:
            Deployment result with rollback information
        """
        return self.deploy_artifact(
            artifact_id=artifact_id,
            target_environment=target_environment,
            strategy=strategy,
            **kwargs,
        )

    def manual_rollback(self, deployment_id: str) -> bool:
        """Manually rollback a deployment.

        Args:
            deployment_id: ID of deployment to rollback

        Returns:
            True if rollback was successful
        """
        return self.orchestrator.manual_rollback(deployment_id)

    def get_deployment_status(self, deployment_id: str) -> dict[str, Any]:
        """Get deployment status and metadata.

        Args:
            deployment_id: Deployment ID

        Returns:
            Deployment status information
        """
        return self.orchestrator.get_deployment_status(deployment_id)

    def get_deployment_history(
        self, artifact_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Get deployment history.

        Args:
            artifact_id: Filter by artifact ID (optional)

        Returns:
            List of deployment statuses
        """
        return self.orchestrator.get_deployment_history(artifact_id)

    def _select_artifact(self, config: DeploymentConfig) -> Any:
        """Select appropriate artifact for deployment.

        Args:
            config: Deployment configuration

        Returns:
            Artifact selection result
        """
        criteria = config.selection_criteria or self._create_default_criteria(
            config
        )
        return self.artifact_selector.select_artifact(criteria)

    def _create_default_criteria(
        self, config: DeploymentConfig
    ) -> SelectionCriteria:
        """Create default selection criteria for deployment.

        Args:
            config: Deployment configuration

        Returns:
            Selection criteria
        """
        return SelectionCriteria(
            min_accuracy=0.8,
            max_inference_time_ms=1000.0,
            max_memory_usage_mb=2048.0,
            max_model_size_mb=100.0,
            preferred_format=config.target_format,
            target_environment=config.target_environment,
            deployment_type=config.deployment_type,
        )

    def _deploy_container(
        self, config: DeploymentConfig, env_config: Any, packaging_result: Any
    ) -> dict[str, Any]:
        """Deploy to container environment.

        Args:
            config: Deployment configuration
            env_config: Environment configuration
            packaging_result: Packaging result

        Returns:
            Deployment result
        """
        self.logger.info("Deploying to container environment")

        # Extract container information from packaging result
        container_image = packaging_result.container_image_name
        dockerfile_path = packaging_result.dockerfile_path

        # Simulate container deployment
        deployment_url = "http://localhost:8501"
        health_check_url = f"{deployment_url}/healthz"

        return {
            "deployment_url": deployment_url,
            "health_check_url": health_check_url,
            "container_image": container_image,
            "dockerfile_path": dockerfile_path,
        }

    def _deploy_kubernetes(
        self, config: DeploymentConfig, env_config: Any, packaging_result: Any
    ) -> dict[str, Any]:
        """Deploy to Kubernetes environment.

        Args:
            config: Deployment configuration
            env_config: Environment configuration
            packaging_result: Packaging result

        Returns:
            Deployment result
        """
        self.logger.info("Deploying to Kubernetes environment")

        # Extract Kubernetes manifests from packaging result
        kubernetes_manifests = packaging_result.kubernetes_manifests
        helm_chart_path = packaging_result.helm_chart_path

        # Simulate Kubernetes deployment
        deployment_url = (
            f"http://crackseg-{config.target_environment}.example.com"
        )
        health_check_url = f"{deployment_url}/healthz"

        return {
            "deployment_url": deployment_url,
            "health_check_url": health_check_url,
            "kubernetes_manifests": kubernetes_manifests,
            "helm_chart_path": helm_chart_path,
        }

    def _deploy_serverless(
        self, config: DeploymentConfig, env_config: Any, packaging_result: Any
    ) -> dict[str, Any]:
        """Deploy to serverless environment.

        Args:
            config: Deployment configuration
            env_config: Environment configuration
            packaging_result: Packaging result

        Returns:
            Deployment result
        """
        self.logger.info("Deploying to serverless environment")

        # Simulate serverless deployment
        deployment_url = f"https://crackseg-{config.target_environment}.lambda.amazonaws.com"
        health_check_url = f"{deployment_url}/healthz"

        return {
            "deployment_url": deployment_url,
            "health_check_url": health_check_url,
            "function_name": f"crackseg-{config.target_environment}",
        }

    def _deploy_edge(
        self, config: DeploymentConfig, env_config: Any, packaging_result: Any
    ) -> dict[str, Any]:
        """Deploy to edge environment.

        Args:
            config: Deployment configuration
            env_config: Environment configuration
            packaging_result: Packaging result

        Returns:
            Deployment result
        """
        self.logger.info("Deploying to edge environment")

        # Simulate edge deployment
        deployment_url = (
            f"http://edge-crackseg-{config.target_environment}.local"
        )
        health_check_url = f"{deployment_url}/healthz"

        return {
            "deployment_url": deployment_url,
            "health_check_url": health_check_url,
            "edge_device_id": f"edge-{config.target_environment}",
        }

    def get_artifact_recommendations(
        self, target_environment: str, deployment_type: str
    ) -> dict[str, Any]:
        """Get artifact recommendations for deployment.

        Args:
            target_environment: Target environment
            deployment_type: Deployment type

        Returns:
            Artifact recommendations
        """
        return self.artifact_selector.get_recommendations(
            target_environment, deployment_type
        )

    def get_environment_summary(self, env_config: Any) -> dict[str, Any]:
        """Get environment configuration summary.

        Args:
            env_config: Environment configuration

        Returns:
            Environment summary
        """
        return self.environment_configurator.get_environment_summary(
            env_config
        )
