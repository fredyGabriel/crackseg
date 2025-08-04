"""File generation for environment configuration."""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import EnvironmentConfig


class ConfigurationFileGenerator:
    """Generator for configuration files."""

    def __init__(self) -> None:
        """Initialize generator."""
        self.base_output_dir = Path("infrastructure/deployment/packages")

    def generate_all_files(self, env_config: "EnvironmentConfig") -> list[str]:
        """Generate all configuration files for environment.

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
        self, env_config: "EnvironmentConfig"
    ) -> Path:
        """Generate requirements.txt file.

        Args:
            env_config: Environment configuration

        Returns:
            Path to generated requirements.txt
        """
        output_dir = self.base_output_dir / env_config.environment_name
        output_dir.mkdir(parents=True, exist_ok=True)

        requirements_path = output_dir / "requirements.txt"
        with open(requirements_path, "w") as f:
            for package in env_config.required_packages:
                f.write(f"{package}\n")

        return requirements_path

    def _generate_dockerfile(self, env_config: "EnvironmentConfig") -> Path:
        """Generate Dockerfile for container deployment.

        Args:
            env_config: Environment configuration

        Returns:
            Path to generated Dockerfile
        """
        output_dir = self.base_output_dir / env_config.environment_name
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
        self, env_config: "EnvironmentConfig"
    ) -> Path:
        """Generate Kubernetes configuration.

        Args:
            env_config: Environment configuration

        Returns:
            Path to generated Kubernetes config
        """
        output_dir = self.base_output_dir / env_config.environment_name
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
        self, env_config: "EnvironmentConfig"
    ) -> Path:
        """Generate serverless configuration.

        Args:
            env_config: Environment configuration

        Returns:
            Path to generated serverless config
        """
        output_dir = self.base_output_dir / env_config.environment_name
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
