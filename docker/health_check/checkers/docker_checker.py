"""Docker container health verification."""

import logging
import subprocess
from typing import Any

from ..models import HealthStatus


class DockerChecker:
    """Docker container health status verification."""

    def __init__(self) -> None:
        """Initialize Docker checker."""
        self.logger = logging.getLogger("docker_checker")

    async def check_container_status(
        self, container_name: str
    ) -> HealthStatus:
        """
        Check Docker container health status. Args: container_name: Name of
        Docker container to check Returns: Docker health status
        """
        try:
            # Check if container exists and is running
            cmd = [
                "docker",
                "inspect",
                "--format",
                "{{.State.Health.Status}}",
                container_name,
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=5
            )

            if result.returncode != 0:
                # Container might not exist or not running
                cmd_status = [
                    "docker",
                    "inspect",
                    "--format",
                    "{{.State.Status}}",
                    container_name,
                ]
                status_result = subprocess.run(
                    cmd_status, capture_output=True, text=True, timeout=5
                )

                if status_result.returncode != 0:
                    return HealthStatus.UNKNOWN

                container_status = status_result.stdout.strip()
                if container_status == "running":
                    return HealthStatus.HEALTHY  # No health check configured
                else:
                    return HealthStatus.UNHEALTHY

            health_status = result.stdout.strip()

            # Map Docker health status to our enum
            status_mapping = {
                "healthy": HealthStatus.HEALTHY,
                "unhealthy": HealthStatus.UNHEALTHY,
                "starting": HealthStatus.STARTING,
                "none": HealthStatus.HEALTHY,  # No health check configured
                "": HealthStatus.UNKNOWN,
            }

            return status_mapping.get(health_status, HealthStatus.UNKNOWN)

        except subprocess.TimeoutExpired:
            self.logger.warning(
                "Docker command timed out for container %s", container_name
            )
            return HealthStatus.UNKNOWN
        except Exception as e:
            self.logger.error(
                "Error checking Docker status for %s: %s", container_name, e
            )
            return HealthStatus.UNKNOWN

    def get_container_info(self, container_name: str) -> dict[str, Any]:
        """
        Get detailed container information. Args: container_name: Name of
        Docker container Returns: Dictionary with container details
        """
        try:
            cmd = [
                "docker",
                "inspect",
                "--format",
                "{{json .}}",
                container_name,
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10
            )

            if result.returncode == 0:
                import json

                return json.loads(result.stdout)
            else:
                return {"error": "Container not found or not accessible"}

        except Exception as e:
            self.logger.error(
                "Failed to get container info for %s: %s", container_name, e
            )
            return {"error": str(e)}
