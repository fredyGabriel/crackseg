"""Multi-target deployment support for CrackSeg.

This module extends the deployment system to support multiple target
environments with specific configurations and validations for each environment.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

# Import from core module to avoid type conflicts
from ..core.manager import DeploymentConfig, DeploymentResult
from ..core.orchestrator import DeploymentOrchestrator, DeploymentStrategy
from .defaults_data import get_default_environment_data
from .env_utils import compute_resource_issues, serialize_environment_configs


class TargetEnvironment(Enum):
    """Supported target environments for deployment."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"
    DEMO = "demo"


@dataclass
class EnvironmentConfig:
    """Configuration for a specific target environment."""

    name: TargetEnvironment
    deployment_strategy: DeploymentStrategy
    health_check_timeout: int = 30
    max_retries: int = 3
    auto_rollback: bool = True
    performance_thresholds: dict[str, float] | None = None
    resource_limits: dict[str, Any] | None = None
    security_requirements: dict[str, Any] | None = None
    monitoring_config: dict[str, Any] | None = None


class MultiTargetDeploymentManager:
    """Manages deployments across multiple target environments."""

    def __init__(self) -> None:
        """Initialize multi-target deployment manager."""
        self.logger = logging.getLogger(__name__)
        self.environment_configs: dict[
            TargetEnvironment, EnvironmentConfig
        ] = {}
        self.orchestrators: dict[TargetEnvironment, DeploymentOrchestrator] = (
            {}
        )
        self._initialize_default_configs()

    def _initialize_default_configs(self) -> None:
        """Initialize default configurations for each environment."""
        raw = get_default_environment_data()
        name_to_env = {
            "development": TargetEnvironment.DEVELOPMENT,
            "staging": TargetEnvironment.STAGING,
            "production": TargetEnvironment.PRODUCTION,
            "testing": TargetEnvironment.TESTING,
            "demo": TargetEnvironment.DEMO,
        }
        # Map enum name strings to actual DeploymentStrategy values
        for key, data in raw.items():
            env = name_to_env[key]
            strategy = DeploymentStrategy[data["deployment_strategy"]]
            self.environment_configs[env] = EnvironmentConfig(
                name=env,
                deployment_strategy=strategy,
                health_check_timeout=data["health_check_timeout"],
                max_retries=data["max_retries"],
                auto_rollback=data["auto_rollback"],
                performance_thresholds=data["performance_thresholds"],
                resource_limits=data["resource_limits"],
                security_requirements=data["security_requirements"],
                monitoring_config=data["monitoring_config"],
            )

    def get_orchestrator(
        self, environment: TargetEnvironment
    ) -> DeploymentOrchestrator:
        """Get or create orchestrator for specific environment.

        Args:
            environment: Target environment

        Returns:
            Deployment orchestrator for the environment
        """
        if environment not in self.orchestrators:
            self.orchestrators[environment] = DeploymentOrchestrator()
            self.logger.info(f"Created orchestrator for {environment.value}")

        return self.orchestrators[environment]

    def get_environment_config(
        self, environment: TargetEnvironment
    ) -> EnvironmentConfig:
        """Get configuration for specific environment.

        Args:
            environment: Target environment

        Returns:
            Environment configuration
        """
        if environment not in self.environment_configs:
            raise ValueError(f"Unknown environment: {environment}")

        return self.environment_configs[environment]

    def deploy_to_environment(
        self,
        config: DeploymentConfig,
        environment: TargetEnvironment,
        deployment_func: Any,
        **kwargs,
    ) -> DeploymentResult:
        """Deploy to specific environment with environment-specific
        configuration.

        Args:
            config: Deployment configuration
            environment: Target environment
            deployment_func: Function to perform actual deployment
            **kwargs: Additional deployment parameters

        Returns:
            Deployment result
        """
        env_config = self.get_environment_config(environment)
        orchestrator = self.get_orchestrator(environment)

        # Update config with environment-specific settings
        config.target_environment = environment.value
        config.deployment_type = "container"  # Default for all environments

        # Add environment-specific parameters
        kwargs.update(
            {
                "max_retries": env_config.max_retries,
                "auto_rollback": env_config.auto_rollback,
                "performance_thresholds": env_config.performance_thresholds,
                "resource_limits": env_config.resource_limits,
                "security_requirements": env_config.security_requirements,
                "monitoring_config": env_config.monitoring_config,
            }
        )

        self.logger.info(
            f"Deploying to {environment.value} with strategy "
            f"{env_config.deployment_strategy.value}"
        )

        # Perform deployment with environment-specific strategy
        result = orchestrator.deploy_with_strategy(
            config=config,
            strategy=env_config.deployment_strategy,
            deployment_func=deployment_func,
            **kwargs,
        )

        return result

    def deploy_to_multiple_environments(
        self,
        config: DeploymentConfig,
        environments: list[TargetEnvironment],
        deployment_func: Any,
        **kwargs,
    ) -> dict[TargetEnvironment, DeploymentResult]:
        """Deploy to multiple environments sequentially.

        Args:
            config: Deployment configuration
            environments: List of target environments
            deployment_func: Function to perform actual deployment
            **kwargs: Additional deployment parameters

        Returns:
            Dictionary mapping environments to deployment results
        """
        results: dict[TargetEnvironment, DeploymentResult] = {}

        for environment in environments:
            try:
                self.logger.info(f"Starting deployment to {environment.value}")
                result = self.deploy_to_environment(
                    config, environment, deployment_func, **kwargs
                )
                results[environment] = result

                if not result.success:
                    self.logger.error(
                        f"Deployment to {environment.value} failed: "
                        f"{result.error_message}"
                    )
                    # Continue with other environments unless critical
                    if environment == TargetEnvironment.PRODUCTION:
                        break

            except Exception as e:
                self.logger.error(
                    f"Error deploying to {environment.value}: {e}"
                )
                results[environment] = DeploymentResult(
                    success=False,
                    deployment_id=f"failed-{environment.value}",
                    message=f"Deployment failed: {str(e)}",
                    error_message=str(e),
                )

        return results

    def validate_environment_readiness(
        self, environment: TargetEnvironment
    ) -> dict[str, Any]:
        """Validate that environment is ready for deployment.

        Args:
            environment: Target environment to validate

        Returns:
            Validation results
        """
        env_config = self.get_environment_config(environment)
        validation_results = {
            "environment": environment.value,
            "ready": True,
            "issues": [],
            "warnings": [],
        }

        # Check resource availability
        issues, warnings = compute_resource_issues(env_config.resource_limits)
        if issues:
            validation_results["issues"].extend(issues)
            validation_results["ready"] = False
        if warnings:
            validation_results["warnings"].extend(warnings)

        # Check security requirements
        if env_config.security_requirements:
            if env_config.security_requirements.get("ssl_required", False):
                # In a real implementation, check SSL certificate availability
                validation_results["warnings"].append(
                    "SSL certificate validation not implemented"
                )

        self.logger.info(
            f"Environment {environment.value} validation: "
            f"{'READY' if validation_results['ready'] else 'NOT READY'}"
        )

        return validation_results

    def get_deployment_status_across_environments(
        self, deployment_id: str
    ) -> dict[TargetEnvironment, dict[str, Any]]:
        """Get deployment status across all environments.

        Args:
            deployment_id: Deployment ID to check

        Returns:
            Dictionary mapping environments to deployment status
        """
        statuses = {}

        for environment in TargetEnvironment:
            try:
                orchestrator = self.get_orchestrator(environment)
                status = orchestrator.get_deployment_status(deployment_id)
                statuses[environment] = status
            except Exception as e:
                self.logger.warning(
                    f"Error getting status for {environment.value}: {e}"
                )
                statuses[environment] = {"error": str(e)}

        return statuses

    def rollback_across_environments(
        self,
        deployment_id: str,
        environments: list[TargetEnvironment] | None = None,
    ) -> dict[TargetEnvironment, bool]:
        """Rollback deployment across multiple environments.

        Args:
            deployment_id: Deployment ID to rollback
            environments: List of environments to rollback (None for all)

        Returns:
            Dictionary mapping environments to rollback success
        """
        if environments is None:
            environments = list(TargetEnvironment)

        results = {}

        for environment in environments:
            try:
                orchestrator = self.get_orchestrator(environment)
                success = orchestrator.manual_rollback(deployment_id)
                results[environment] = success

                if success:
                    self.logger.info(
                        f"Rollback successful for {environment.value}"
                    )
                else:
                    self.logger.error(
                        f"Rollback failed for {environment.value}"
                    )

            except Exception as e:
                self.logger.error(
                    f"Error during rollback for {environment.value}: {e}"
                )
                results[environment] = False

        return results

    def export_environment_configs(self, output_path: Path) -> None:
        """Export environment configurations to file.

        Args:
            output_path: Path to export configurations
        """
        import json

        data = serialize_environment_configs(self.environment_configs)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        self.logger.info(f"Exported environment configs to {output_path}")
