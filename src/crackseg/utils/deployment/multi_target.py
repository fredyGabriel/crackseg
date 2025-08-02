"""Multi-target deployment support for CrackSeg.

This module extends the deployment system to support multiple target
environments with specific configurations and validations for each environment.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .config import DeploymentConfig, DeploymentResult
from .orchestration import DeploymentOrchestrator, DeploymentStrategy


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
        # Development environment
        self.environment_configs[TargetEnvironment.DEVELOPMENT] = (
            EnvironmentConfig(
                name=TargetEnvironment.DEVELOPMENT,
                deployment_strategy=DeploymentStrategy.RECREATE,
                health_check_timeout=10,
                max_retries=1,
                auto_rollback=False,
                performance_thresholds={
                    "response_time_ms": 1000,
                    "memory_usage_mb": 2048,
                    "cpu_usage_percent": 80,
                },
                resource_limits={
                    "memory_mb": 2048,
                    "cpu_cores": 2,
                    "disk_gb": 10,
                },
                security_requirements={
                    "ssl_required": False,
                    "authentication_required": False,
                },
                monitoring_config={
                    "check_interval": 60,
                    "alert_threshold": 0.8,
                },
            )
        )

        # Staging environment
        self.environment_configs[TargetEnvironment.STAGING] = (
            EnvironmentConfig(
                name=TargetEnvironment.STAGING,
                deployment_strategy=DeploymentStrategy.BLUE_GREEN,
                health_check_timeout=30,
                max_retries=2,
                auto_rollback=True,
                performance_thresholds={
                    "response_time_ms": 500,
                    "memory_usage_mb": 4096,
                    "cpu_usage_percent": 70,
                },
                resource_limits={
                    "memory_mb": 4096,
                    "cpu_cores": 4,
                    "disk_gb": 20,
                },
                security_requirements={
                    "ssl_required": True,
                    "authentication_required": True,
                },
                monitoring_config={
                    "check_interval": 30,
                    "alert_threshold": 0.9,
                },
            )
        )

        # Production environment
        self.environment_configs[TargetEnvironment.PRODUCTION] = (
            EnvironmentConfig(
                name=TargetEnvironment.PRODUCTION,
                deployment_strategy=DeploymentStrategy.CANARY,
                health_check_timeout=60,
                max_retries=3,
                auto_rollback=True,
                performance_thresholds={
                    "response_time_ms": 200,
                    "memory_usage_mb": 8192,
                    "cpu_usage_percent": 60,
                },
                resource_limits={
                    "memory_mb": 8192,
                    "cpu_cores": 8,
                    "disk_gb": 50,
                },
                security_requirements={
                    "ssl_required": True,
                    "authentication_required": True,
                    "encryption_required": True,
                },
                monitoring_config={
                    "check_interval": 15,
                    "alert_threshold": 0.95,
                },
            )
        )

        # Testing environment
        self.environment_configs[TargetEnvironment.TESTING] = (
            EnvironmentConfig(
                name=TargetEnvironment.TESTING,
                deployment_strategy=DeploymentStrategy.RECREATE,
                health_check_timeout=15,
                max_retries=1,
                auto_rollback=False,
                performance_thresholds={
                    "response_time_ms": 2000,
                    "memory_usage_mb": 1024,
                    "cpu_usage_percent": 90,
                },
                resource_limits={
                    "memory_mb": 1024,
                    "cpu_cores": 1,
                    "disk_gb": 5,
                },
                security_requirements={
                    "ssl_required": False,
                    "authentication_required": False,
                },
                monitoring_config={
                    "check_interval": 120,
                    "alert_threshold": 0.5,
                },
            )
        )

        # Demo environment
        self.environment_configs[TargetEnvironment.DEMO] = EnvironmentConfig(
            name=TargetEnvironment.DEMO,
            deployment_strategy=DeploymentStrategy.ROLLING,
            health_check_timeout=20,
            max_retries=2,
            auto_rollback=True,
            performance_thresholds={
                "response_time_ms": 800,
                "memory_usage_mb": 3072,
                "cpu_usage_percent": 75,
            },
            resource_limits={
                "memory_mb": 3072,
                "cpu_cores": 2,
                "disk_gb": 15,
            },
            security_requirements={
                "ssl_required": True,
                "authentication_required": False,
            },
            monitoring_config={
                "check_interval": 45,
                "alert_threshold": 0.8,
            },
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
                    artifact_id=config.artifact_id,
                    target_environment=environment.value,
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
        try:
            import psutil

            # Check system resources
            memory = psutil.virtual_memory()
            cpu_count = psutil.cpu_count()
            disk = psutil.disk_usage("/")

            if env_config.resource_limits:
                if (
                    memory.total
                    < env_config.resource_limits.get("memory_mb", 0)
                    * 1024
                    * 1024
                ):
                    memory_gb = memory.total / (1024**3)
                    validation_results["issues"].append(
                        f"Insufficient memory: {memory_gb:.1f}GB available, "
                        f"{env_config.resource_limits['memory_mb']}MB required"
                    )
                    validation_results["ready"] = False

                if cpu_count < env_config.resource_limits.get("cpu_cores", 0):
                    validation_results["issues"].append(
                        f"Insufficient CPU cores: {cpu_count} available, "
                        f"{env_config.resource_limits['cpu_cores']} required"
                    )
                    validation_results["ready"] = False

                if (
                    disk.free
                    < env_config.resource_limits.get("disk_gb", 0)
                    * 1024
                    * 1024
                    * 1024
                ):
                    disk_gb = disk.free / (1024**3)
                    validation_results["issues"].append(
                        f"Insufficient disk space: {disk_gb:.1f}GB available, "
                        f"{env_config.resource_limits['disk_gb']}GB required"
                    )
                    validation_results["ready"] = False

        except ImportError:
            validation_results["warnings"].append(
                "psutil not available, skipping resource validation"
            )

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

        configs_dict = {}
        for env, config in self.environment_configs.items():
            configs_dict[env.value] = {
                "deployment_strategy": config.deployment_strategy.value,
                "health_check_timeout": config.health_check_timeout,
                "max_retries": config.max_retries,
                "auto_rollback": config.auto_rollback,
                "performance_thresholds": config.performance_thresholds,
                "resource_limits": config.resource_limits,
                "security_requirements": config.security_requirements,
                "monitoring_config": config.monitoring_config,
            }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(configs_dict, f, indent=2)

        self.logger.info(f"Exported environment configs to {output_path}")
