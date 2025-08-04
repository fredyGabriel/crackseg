"""Core environment configurator implementation."""

import logging
from typing import TYPE_CHECKING, Any

from .config import ConfigurationResult, EnvironmentConfig
from .generators import ConfigurationFileGenerator
from .presets import load_predefined_configs

if TYPE_CHECKING:
    from .config import DeploymentConfig


class EnvironmentConfigurator:
    """Environment configurator for deployment.

    Handles environment configuration for different deployment targets,
    including resource requirements, dependencies, and deployment-specific
    settings.
    """

    def __init__(self) -> None:
        """Initialize environment configurator."""
        self.logger = logging.getLogger(__name__)
        self.supported_environments = ["production", "staging", "development"]
        self.supported_targets = ["container", "serverless", "edge"]

        # Predefined configurations
        self.predefined_configs = load_predefined_configs()

        # Initialize components
        self.generator = ConfigurationFileGenerator()

        self.logger.info("EnvironmentConfigurator initialized")

    def configure_environment(
        self, deployment_config: "DeploymentConfig"
    ) -> ConfigurationResult:
        """Configure environment for deployment.

        Args:
            deployment_config: Deployment configuration

        Returns:
            Configuration result
        """
        self.logger.info(
            f"Configuring environment for "
            f"{deployment_config.target_environment} "
            f"with {deployment_config.deployment_type} deployment"
        )

        try:
            # Validate environment and target
            validation_errors = self._validate_configuration(deployment_config)
            if validation_errors:
                return ConfigurationResult(
                    success=False,
                    validation_errors=validation_errors,
                    error_message="Configuration validation failed",
                )

            # Get predefined configuration
            env_config = self._get_predefined_config(deployment_config)

            # Customize configuration based on deployment config
            env_config = self._customize_configuration(
                env_config, deployment_config
            )

            # Generate configuration files
            config_files = self.generator.generate_all_files(env_config)

            return ConfigurationResult(
                success=True,
                environment_config=env_config,
                configuration_files=config_files,
            )

        except Exception as e:
            self.logger.error(f"Environment configuration failed: {e}")
            return ConfigurationResult(
                success=False,
                error_message=str(e),
            )

    def _validate_configuration(
        self, deployment_config: "DeploymentConfig"
    ) -> list[str]:
        """Validate deployment configuration.

        Args:
            deployment_config: Deployment configuration

        Returns:
            List of validation errors
        """
        errors = []

        # Validate environment
        if (
            deployment_config.target_environment
            not in self.supported_environments
        ):
            errors.append(
                f"Unsupported environment: "
                f"{deployment_config.target_environment}"
            )

        # Validate deployment type
        if deployment_config.deployment_type not in self.supported_targets:
            errors.append(
                f"Unsupported deployment type: "
                f"{deployment_config.deployment_type}"
            )

        return errors

    def _get_predefined_config(
        self, deployment_config: "DeploymentConfig"
    ) -> EnvironmentConfig:
        """Get predefined configuration for environment and target.

        Args:
            deployment_config: Deployment configuration

        Returns:
            Environment configuration
        """
        env_name = deployment_config.target_environment
        target = deployment_config.deployment_type

        if env_name not in self.predefined_configs:
            raise ValueError(f"Unknown environment: {env_name}")

        if target not in self.predefined_configs[env_name]:
            raise ValueError(
                f"Unknown target: {target} for environment: {env_name}"
            )

        config_data = self.predefined_configs[env_name][target]

        return EnvironmentConfig(
            environment_name=f"{env_name}-{target}",
            environment_type=env_name,
            deployment_target=target,
            resources=config_data["resources"],
            python_version=config_data["python_version"],
            required_packages=config_data["required_packages"],
            system_dependencies=config_data.get("system_dependencies", []),
            base_image=config_data.get("base_image", "python:3.9-slim"),
            replicas=config_data.get("replicas", 1),
            autoscaling=config_data.get("autoscaling", False),
            max_replicas=config_data.get("max_replicas", 5),
            log_level=config_data.get("log_level", "INFO"),
        )

    def _customize_configuration(
        self,
        env_config: EnvironmentConfig,
        deployment_config: "DeploymentConfig",
    ) -> EnvironmentConfig:
        """Customize configuration based on deployment config.

        Args:
            env_config: Base environment configuration
            deployment_config: Deployment configuration

        Returns:
            Customized environment configuration
        """
        # Add deployment-specific packages
        if deployment_config.enable_quantization:
            env_config.required_packages.extend(
                [
                    "onnx>=1.12.0",
                    "onnxruntime>=1.12.0",
                ]
            )

        if deployment_config.target_format == "tensorrt":
            env_config.required_packages.extend(
                [
                    "tensorrt>=8.0.0",
                ]
            )

        # Add environment variables
        env_config.environment_variables.update(
            {
                "DEPLOYMENT_ENVIRONMENT": deployment_config.target_environment,
                "DEPLOYMENT_TYPE": deployment_config.deployment_type,
                "ENABLE_QUANTIZATION": str(
                    deployment_config.enable_quantization
                ),
                "TARGET_FORMAT": deployment_config.target_format,
                "LOG_LEVEL": env_config.log_level,
            }
        )

        # Add health check and metrics endpoints
        env_config.health_check_path = "/healthz"
        env_config.metrics_endpoint = "/metrics"

        # Add security context for production
        if deployment_config.target_environment == "production":
            env_config.security_context = {
                "runAsNonRoot": True,
                "runAsUser": 1000,
                "fsGroup": 1000,
            }

        return env_config

    def get_environment_summary(
        self, env_config: EnvironmentConfig
    ) -> dict[str, Any]:
        """Get summary of environment configuration.

        Args:
            env_config: Environment configuration

        Returns:
            Environment summary
        """
        return {
            "environment_name": env_config.environment_name,
            "environment_type": env_config.environment_type,
            "deployment_target": env_config.deployment_target,
            "resources": {
                "cpu_cores": env_config.resources.cpu_cores,
                "memory_mb": env_config.resources.memory_mb,
                "gpu_memory_mb": env_config.resources.gpu_memory_mb,
                "storage_gb": env_config.resources.storage_gb,
            },
            "python_version": env_config.python_version,
            "required_packages_count": len(env_config.required_packages),
            "system_dependencies_count": len(env_config.system_dependencies),
            "replicas": env_config.replicas,
            "autoscaling": env_config.autoscaling,
            "log_level": env_config.log_level,
        }
