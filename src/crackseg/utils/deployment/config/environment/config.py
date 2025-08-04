"""Configuration dataclasses for environment configuration."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ResourceRequirements:
    """Resource requirements for deployment."""

    cpu_cores: float = 1.0
    memory_mb: int = 1024
    gpu_memory_mb: int = 0
    storage_gb: float = 1.0
    network_bandwidth_mbps: int = 100


@dataclass
class EnvironmentConfig:
    """Environment configuration for deployment."""

    # Environment identification
    environment_name: str
    environment_type: str  # "production", "staging", "development"
    deployment_target: str  # "container", "serverless", "edge"

    # Resource requirements
    resources: ResourceRequirements = field(
        default_factory=ResourceRequirements
    )

    # Dependencies and packages
    python_version: str = "3.9"
    required_packages: list[str] = field(default_factory=list)
    system_dependencies: list[str] = field(default_factory=list)

    # Deployment-specific settings
    container_image: str | None = None
    base_image: str = "python:3.9-slim"
    exposed_ports: list[int] = field(default_factory=lambda: [8501])
    environment_variables: dict[str, str] = field(default_factory=dict)

    # Security and access
    security_context: dict[str, Any] = field(default_factory=dict)
    access_control: dict[str, Any] = field(default_factory=dict)

    # Monitoring and logging
    health_check_path: str = "/healthz"
    metrics_endpoint: str = "/metrics"
    log_level: str = "INFO"

    # Scaling and performance
    replicas: int = 1
    autoscaling: bool = False
    max_replicas: int = 5
    min_replicas: int = 1

    # Additional configuration
    custom_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfigurationResult:
    """Result of environment configuration."""

    success: bool
    environment_config: EnvironmentConfig | None = None
    configuration_files: list[str] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error_message: str | None = None
