"""Environment configuration for deployment.

This module provides environment configuration capabilities for different
deployment targets, including resource requirements, dependencies, and
deployment-specific settings.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import DeploymentConfig


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
        self._load_predefined_configs()

        self.logger.info("EnvironmentConfigurator initialized")

    def _load_predefined_configs(self) -> None:
        """Load predefined environment configurations."""
        self.predefined_configs = {
            "production": {
                "container": {
                    "resources": ResourceRequirements(
                        cpu_cores=4.0,
                        memory_mb=8192,
                        gpu_memory_mb=8192,
                        storage_gb=10.0,
                        network_bandwidth_mbps=1000,
                    ),
                    "python_version": "3.9",
                    "base_image": "python:3.9-slim",
                    "required_packages": [
                        "torch>=1.9.0",
                        "torchvision>=0.10.0",
                        "streamlit>=1.0.0",
                        "opencv-python>=4.5.0",
                        "pillow>=8.0.0",
                        "numpy>=1.21.0",
                    ],
                    "system_dependencies": [
                        "libgl1-mesa-glx",
                        "libglib2.0-0",
                        "libsm6",
                        "libxext6",
                        "libxrender-dev",
                    ],
                    "replicas": 3,
                    "autoscaling": True,
                    "max_replicas": 10,
                    "log_level": "WARNING",
                },
                "serverless": {
                    "resources": ResourceRequirements(
                        cpu_cores=2.0,
                        memory_mb=4096,
                        gpu_memory_mb=0,
                        storage_gb=1.0,
                        network_bandwidth_mbps=100,
                    ),
                    "python_version": "3.9",
                    "required_packages": [
                        "torch>=1.9.0",
                        "streamlit>=1.0.0",
                        "opencv-python>=4.5.0",
                        "pillow>=8.0.0",
                        "numpy>=1.21.0",
                    ],
                    "replicas": 1,
                    "autoscaling": False,
                    "log_level": "INFO",
                },
                "edge": {
                    "resources": ResourceRequirements(
                        cpu_cores=1.0,
                        memory_mb=2048,
                        gpu_memory_mb=0,
                        storage_gb=2.0,
                        network_bandwidth_mbps=50,
                    ),
                    "python_version": "3.8",
                    "required_packages": [
                        "torch>=1.8.0",
                        "opencv-python>=4.5.0",
                        "pillow>=8.0.0",
                        "numpy>=1.19.0",
                    ],
                    "replicas": 1,
                    "autoscaling": False,
                    "log_level": "ERROR",
                },
            },
            "staging": {
                "container": {
                    "resources": ResourceRequirements(
                        cpu_cores=2.0,
                        memory_mb=4096,
                        gpu_memory_mb=4096,
                        storage_gb=5.0,
                        network_bandwidth_mbps=500,
                    ),
                    "python_version": "3.9",
                    "base_image": "python:3.9-slim",
                    "required_packages": [
                        "torch>=1.9.0",
                        "torchvision>=0.10.0",
                        "streamlit>=1.0.0",
                        "opencv-python>=4.5.0",
                        "pillow>=8.0.0",
                        "numpy>=1.21.0",
                    ],
                    "system_dependencies": [
                        "libgl1-mesa-glx",
                        "libglib2.0-0",
                        "libsm6",
                        "libxext6",
                        "libxrender-dev",
                    ],
                    "replicas": 2,
                    "autoscaling": True,
                    "max_replicas": 5,
                    "log_level": "INFO",
                },
                "serverless": {
                    "resources": ResourceRequirements(
                        cpu_cores=1.0,
                        memory_mb=2048,
                        gpu_memory_mb=0,
                        storage_gb=1.0,
                        network_bandwidth_mbps=100,
                    ),
                    "python_version": "3.9",
                    "required_packages": [
                        "torch>=1.9.0",
                        "streamlit>=1.0.0",
                        "opencv-python>=4.5.0",
                        "pillow>=8.0.0",
                        "numpy>=1.21.0",
                    ],
                    "replicas": 1,
                    "autoscaling": False,
                    "log_level": "INFO",
                },
                "edge": {
                    "resources": ResourceRequirements(
                        cpu_cores=1.0,
                        memory_mb=1024,
                        gpu_memory_mb=0,
                        storage_gb=1.0,
                        network_bandwidth_mbps=25,
                    ),
                    "python_version": "3.8",
                    "required_packages": [
                        "torch>=1.8.0",
                        "opencv-python>=4.5.0",
                        "pillow>=8.0.0",
                        "numpy>=1.19.0",
                    ],
                    "replicas": 1,
                    "autoscaling": False,
                    "log_level": "WARNING",
                },
            },
            "development": {
                "container": {
                    "resources": ResourceRequirements(
                        cpu_cores=1.0,
                        memory_mb=2048,
                        gpu_memory_mb=2048,
                        storage_gb=2.0,
                        network_bandwidth_mbps=100,
                    ),
                    "python_version": "3.9",
                    "base_image": "python:3.9-slim",
                    "required_packages": [
                        "torch>=1.9.0",
                        "torchvision>=0.10.0",
                        "streamlit>=1.0.0",
                        "opencv-python>=4.5.0",
                        "pillow>=8.0.0",
                        "numpy>=1.21.0",
                        "pytest>=6.0.0",
                        "black>=21.0.0",
                        "ruff>=0.1.0",
                    ],
                    "system_dependencies": [
                        "libgl1-mesa-glx",
                        "libglib2.0-0",
                        "libsm6",
                        "libxext6",
                        "libxrender-dev",
                    ],
                    "replicas": 1,
                    "autoscaling": False,
                    "log_level": "DEBUG",
                },
                "serverless": {
                    "resources": ResourceRequirements(
                        cpu_cores=1.0,
                        memory_mb=1024,
                        gpu_memory_mb=0,
                        storage_gb=1.0,
                        network_bandwidth_mbps=50,
                    ),
                    "python_version": "3.9",
                    "required_packages": [
                        "torch>=1.9.0",
                        "streamlit>=1.0.0",
                        "opencv-python>=4.5.0",
                        "pillow>=8.0.0",
                        "numpy>=1.21.0",
                    ],
                    "replicas": 1,
                    "autoscaling": False,
                    "log_level": "DEBUG",
                },
                "edge": {
                    "resources": ResourceRequirements(
                        cpu_cores=1.0,
                        memory_mb=512,
                        gpu_memory_mb=0,
                        storage_gb=0.5,
                        network_bandwidth_mbps=10,
                    ),
                    "python_version": "3.8",
                    "required_packages": [
                        "torch>=1.8.0",
                        "opencv-python>=4.5.0",
                        "pillow>=8.0.0",
                        "numpy>=1.19.0",
                    ],
                    "replicas": 1,
                    "autoscaling": False,
                    "log_level": "INFO",
                },
            },
        }

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
            config_files = self._generate_configuration_files(env_config)

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

    def _generate_configuration_files(
        self, env_config: EnvironmentConfig
    ) -> list[str]:
        """Generate configuration files for environment.

        Args:
            env_config: Environment configuration

        Returns:
            List of generated configuration file paths
        """
        config_files = []

        # Generate requirements.txt
        requirements_path = self._generate_requirements_file(env_config)
        config_files.append(str(requirements_path))

        # Generate environment-specific config
        if env_config.deployment_target == "container":
            dockerfile_path = self._generate_dockerfile(env_config)
            config_files.append(str(dockerfile_path))

            k8s_path = self._generate_kubernetes_config(env_config)
            config_files.append(str(k8s_path))

        elif env_config.deployment_target == "serverless":
            serverless_path = self._generate_serverless_config(env_config)
            config_files.append(str(serverless_path))

        return config_files

    def _generate_requirements_file(
        self, env_config: EnvironmentConfig
    ) -> Path:
        """Generate requirements.txt file.

        Args:
            env_config: Environment configuration

        Returns:
            Path to generated requirements.txt
        """
        output_dir = Path(f"deployments/{env_config.environment_name}")
        output_dir.mkdir(parents=True, exist_ok=True)

        requirements_path = output_dir / "requirements.txt"
        with open(requirements_path, "w") as f:
            for package in env_config.required_packages:
                f.write(f"{package}\n")

        return requirements_path

    def _generate_dockerfile(self, env_config: EnvironmentConfig) -> Path:
        """Generate Dockerfile for container deployment.

        Args:
            env_config: Environment configuration

        Returns:
            Path to generated Dockerfile
        """
        output_dir = Path(f"deployments/{env_config.environment_name}")
        output_dir.mkdir(parents=True, exist_ok=True)

        dockerfile_path = output_dir / "Dockerfile"
        with open(dockerfile_path, "w") as f:
            f.write(f"FROM {env_config.base_image}\n\n")
            f.write("# Set environment variables\n")
            for key, value in env_config.environment_variables.items():
                f.write(f'ENV {key}="{value}"\n')

            f.write("\n# Install system dependencies\n")
            if env_config.system_dependencies:
                f.write("RUN apt-get update && apt-get install -y \\\n")
                for dep in env_config.system_dependencies:
                    f.write(f"    {dep} \\\n")
                f.write("    && rm -rf /var/lib/apt/lists/*\n\n")

            f.write("# Set working directory\n")
            f.write("WORKDIR /app\n\n")

            f.write("# Copy requirements and install Python dependencies\n")
            f.write("COPY requirements.txt .\n")
            f.write("RUN pip install --no-cache-dir -r requirements.txt\n\n")

            f.write("# Copy application code\n")
            f.write("COPY . .\n\n")

            f.write("# Expose port\n")
            for port in env_config.exposed_ports:
                f.write(f"EXPOSE {port}\n")

            f.write("\n# Health check\n")
            f.write(
                "HEALTHCHECK --interval=30s --timeout=3s "
                "--start-period=5s --retries=3 \\\n"
            )
            f.write(
                f"  CMD curl -f {env_config.health_check_path} || exit 1\n\n"
            )

            f.write("# Run application\n")
            f.write(
                'CMD ["streamlit", "run", "app/main.py", '
                '"--server.port=8501", "--server.address=0.0.0.0"]\n'
            )

        return dockerfile_path

    def _generate_kubernetes_config(
        self, env_config: EnvironmentConfig
    ) -> Path:
        """Generate Kubernetes configuration.

        Args:
            env_config: Environment configuration

        Returns:
            Path to generated Kubernetes config
        """
        output_dir = Path(f"deployments/{env_config.environment_name}")
        output_dir.mkdir(parents=True, exist_ok=True)

        k8s_path = output_dir / "k8s-deployment.yaml"
        with open(k8s_path, "w") as f:
            f.write("apiVersion: apps/v1\n")
            f.write("kind: Deployment\n")
            f.write("metadata:\n")
            f.write(f"  name: {env_config.environment_name}\n")
            f.write("spec:\n")
            f.write(f"  replicas: {env_config.replicas}\n")
            f.write("  selector:\n")
            f.write("    matchLabels:\n")
            f.write(f"      app: {env_config.environment_name}\n")
            f.write("  template:\n")
            f.write("    metadata:\n")
            f.write("      labels:\n")
            f.write(f"        app: {env_config.environment_name}\n")
            f.write("    spec:\n")
            f.write("      containers:\n")
            f.write("      - name: app\n")
            f.write(f"        image: {env_config.environment_name}:latest\n")
            f.write("        ports:\n")
            for port in env_config.exposed_ports:
                f.write(f"        - containerPort: {port}\n")
            f.write("        resources:\n")
            f.write("          requests:\n")
            f.write(f"            cpu: {env_config.resources.cpu_cores}\n")
            f.write(
                f"            memory: {env_config.resources.memory_mb}Mi\n"
            )
            f.write("          limits:\n")
            f.write(f"            cpu: {env_config.resources.cpu_cores * 2}\n")
            f.write(
                f"            memory: {env_config.resources.memory_mb * 2}Mi\n"
            )
            f.write("        livenessProbe:\n")
            f.write("          httpGet:\n")
            f.write(f"            path: {env_config.health_check_path}\n")
            f.write("            port: 8501\n")
            f.write("          initialDelaySeconds: 30\n")
            f.write("          periodSeconds: 10\n")

        return k8s_path

    def _generate_serverless_config(
        self, env_config: EnvironmentConfig
    ) -> Path:
        """Generate serverless configuration.

        Args:
            env_config: Environment configuration

        Returns:
            Path to generated serverless config
        """
        output_dir = Path(f"deployments/{env_config.environment_name}")
        output_dir.mkdir(parents=True, exist_ok=True)

        serverless_path = output_dir / "serverless.yml"
        with open(serverless_path, "w") as f:
            f.write("service: crackseg-deployment\n\n")
            f.write("provider:\n")
            f.write("  name: aws\n")
            f.write("  runtime: python3.9\n")
            f.write("  region: us-east-1\n")
            f.write("  memorySize: 2048\n")
            f.write("  timeout: 30\n\n")
            f.write("functions:\n")
            f.write(f"  {env_config.environment_name}:\n")
            f.write("    handler: app.main.handler\n")
            f.write("    events:\n")
            f.write("      - http:\n")
            f.write("          path: /\n")
            f.write("          method: any\n")

        return serverless_path

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
