"""Validation logic for environment configuration."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import DeploymentConfig
    from .config import EnvironmentConfig


class EnvironmentValidator:
    """Validator for environment configurations."""

    def __init__(self) -> None:
        """Initialize validator."""
        self.supported_environments = ["production", "staging", "development"]
        self.supported_targets = ["container", "serverless", "edge"]

    def validate_configuration(
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

    def validate_environment_config(
        self, env_config: EnvironmentConfig
    ) -> list[str]:
        """Validate environment configuration.

        Args:
            env_config: Environment configuration

        Returns:
            List of validation errors
        """
        errors = []

        # Validate resource requirements
        if env_config.resources.cpu_cores <= 0:
            errors.append("CPU cores must be positive")

        if env_config.resources.memory_mb <= 0:
            errors.append("Memory must be positive")

        if env_config.resources.gpu_memory_mb < 0:
            errors.append("GPU memory cannot be negative")

        # Validate scaling configuration
        if env_config.autoscaling:
            if env_config.max_replicas < env_config.min_replicas:
                errors.append(
                    "Max replicas must be greater than or equal to min replicas"
                )

        # Validate ports
        for port in env_config.exposed_ports:
            if not (1 <= port <= 65535):
                errors.append(f"Invalid port number: {port}")

        return errors
