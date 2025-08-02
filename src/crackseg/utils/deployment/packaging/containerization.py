"""Containerization management for packaging system.

This module handles Docker container creation, optimization, and
registry operations.
"""

import logging
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..config import DeploymentConfig

logger = logging.getLogger(__name__)


class ContainerizationManager:
    """Manages containerization operations for packaging."""

    def __init__(self, packaging_system: Any) -> None:
        """Initialize containerization manager.

        Args:
            packaging_system: Reference to main packaging system
        """
        self.packaging_system = packaging_system
        self.logger = logging.getLogger(__name__)

    def create_advanced_dockerfile(
        self,
        package_dir: Path,
        requirements_path: Path,
        config: "DeploymentConfig",
    ) -> Path:
        """Create advanced multi-stage Dockerfile.

        Args:
            package_dir: Package directory
            requirements_path: Path to requirements.txt
            config: Deployment configuration

        Returns:
            Path to created Dockerfile
        """
        dockerfile_path = package_dir / "Dockerfile"

        # Get container configuration
        container_config = self.packaging_system.container_configs.get(
            config.target_environment,
            self.packaging_system.container_configs["development"],
        )

        # Build multi-stage Dockerfile
        dockerfile_content = self._build_dockerfile_content(
            container_config, config
        )

        dockerfile_path.write_text(dockerfile_content)
        self.logger.info(f"Created advanced Dockerfile: {dockerfile_path}")

        return dockerfile_path

    def _build_dockerfile_content(
        self, container_config: Any, config: "DeploymentConfig"
    ) -> str:
        """Build Dockerfile content based on configuration.

        Args:
            container_config: Container configuration
            config: Deployment configuration

        Returns:
            Dockerfile content as string
        """
        base_image = container_config.base_image

        # Multi-stage build for production
        if (
            container_config.multi_stage
            and config.target_environment == "production"
        ):
            return f"""# Multi-stage build for production
FROM {base_image} AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Production stage
FROM {base_image} AS production

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages \\
    /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY config/ ./config/

# Set ownership
RUN chown -R crackseg:crackseg /app

# Switch to non-root user
USER crackseg

# Expose ports
EXPOSE 8501 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
  CMD python scripts/health_check.py

# Start application
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0",
     "--port", "8501"]
"""

        # Single-stage build for development/staging
        else:
            return f"""# Single-stage build for {config.target_environment}
FROM {base_image}

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Create user if needed
{
                "RUN groupadd -r appuser && useradd -r -g appuser appuser"
                if container_config.non_root_user
                else ""
            }

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY config/ ./config/

# Set ownership if using non-root user
{"RUN chown -R appuser:appuser /app" if container_config.non_root_user else ""}

# Switch to non-root user if configured
{"USER appuser" if container_config.non_root_user else ""}

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8501/healthz || exit 1

# Start application
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0",
     "--port", "8501"]
"""

    def build_optimized_container_image(
        self,
        package_dir: Path,
        dockerfile_path: Path,
        config: "DeploymentConfig",
    ) -> dict[str, Any]:
        """Build optimized container image.

        Args:
            package_dir: Package directory
            dockerfile_path: Path to Dockerfile
            config: Deployment configuration

        Returns:
            Dictionary with container build information
        """
        start_time = time.time()

        # Generate image name
        artifact_id = f"crackseg-{config.target_environment}"
        image_name = f"{artifact_id}:latest"

        try:
            # Build Docker image
            build_cmd = [
                "docker",
                "build",
                "-t",
                image_name,
                "-f",
                str(dockerfile_path),
                str(package_dir),
            ]

            # Add build optimizations
            if config.target_environment == "production":
                build_cmd.extend(["--no-cache", "--compress"])

            self.logger.info(f"Building container image: {image_name}")

            # Execute build command
            result = subprocess.run(
                build_cmd, capture_output=True, text=True, check=True
            )

            build_time = time.time() - start_time

            # Get image information
            image_info = self._get_image_info(image_name)

            return {
                "image_name": image_name,
                "image_size_mb": image_info.get("size_mb", 0.0),
                "layer_count": image_info.get("layer_count", 0),
                "build_time_seconds": build_time,
                "build_output": result.stdout,
            }

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Container build failed: {e.stderr}")
            raise RuntimeError(f"Container build failed: {e.stderr}") from e

    def push_to_registry(
        self, image_name: str, config: "DeploymentConfig"
    ) -> None:
        """Push container image to registry.

        Args:
            image_name: Name of the container image
            config: Deployment configuration
        """
        if config.target_environment != "production":
            self.logger.info(
                "Skipping registry push for non-production environment"
            )
            return

        try:
            # Tag for registry
            registry_url = "your-registry.com"  # Configure as needed
            registry_image = f"{registry_url}/{image_name}"

            # Tag image
            subprocess.run(
                ["docker", "tag", image_name, registry_image], check=True
            )

            # Push to registry
            self.logger.info(f"Pushing image to registry: {registry_image}")
            subprocess.run(["docker", "push", registry_image], check=True)

            self.logger.info("Successfully pushed image to registry")

        except subprocess.CalledProcessError as e:
            self.logger.warning(f"Registry push failed: {e}")
            # Don't fail the entire process for registry issues

    def _get_image_info(self, image_name: str) -> dict[str, Any]:
        """Get information about Docker image.

        Args:
            image_name: Name of the Docker image

        Returns:
            Dictionary with image information
        """
        try:
            # Get image size
            size_cmd = [
                "docker",
                "images",
                "--format",
                "{{.Size}}",
                image_name,
            ]
            size_result = subprocess.run(
                size_cmd, capture_output=True, text=True
            )

            if size_result.returncode == 0:
                size_str = size_result.stdout.strip()
                size_mb = self._parse_docker_size(size_str)
            else:
                size_mb = 0.0

            # Get layer count (simplified)
            layer_count = 10  # Default estimate

            return {
                "size_mb": size_mb,
                "layer_count": layer_count,
            }

        except Exception as e:
            self.logger.warning(f"Failed to get image info: {e}")
            return {
                "size_mb": 0.0,
                "layer_count": 0,
            }

    def _parse_docker_size(self, size_str: str) -> float:
        """Parse Docker size string to MB.

        Args:
            size_str: Docker size string (e.g., "1.2GB", "500MB")

        Returns:
            Size in MB as float
        """
        try:
            if "GB" in size_str:
                return float(size_str.replace("GB", "")) * 1024
            elif "MB" in size_str:
                return float(size_str.replace("MB", ""))
            elif "KB" in size_str:
                return float(size_str.replace("KB", "")) / 1024
            else:
                return 0.0
        except (ValueError, AttributeError):
            return 0.0
