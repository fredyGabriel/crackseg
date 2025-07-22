"""Service configuration registry and loader."""

import json
import logging
from pathlib import Path
from typing import Any, cast

from ..models import ServiceConfig


class ServiceRegistry:
    """Service configuration registry and management."""

    def __init__(self, config_path: Path | None = None) -> None:
        """
        Initialize service registry. Args: config_path: Path to health check
        configuration file
        """
        self.logger = logging.getLogger("service_registry")
        self.config_path = (
            config_path
            or Path(__file__).parent.parent.parent / "health_config.json"
        )
        self.services: dict[str, ServiceConfig] = {}

    def load_configuration(self) -> dict[str, ServiceConfig]:
        """
        Load service configuration from file. Returns: Dictionary of service
        configurations
        """
        default_config = self._get_default_configuration()

        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    loaded_config: dict[str, Any] = json.load(f)
                    default_config.update(loaded_config)
            except Exception as e:
                self.logger.warning(
                    "Failed to load config from %s: %s", self.config_path, e
                )

        # Convert to ServiceConfig objects
        self.services = {}
        for name, config in default_config.items():
            self.services[name] = ServiceConfig(
                name=cast(str, config["name"]),
                container_name=cast(str, config["container_name"]),
                health_endpoint=cast(str, config["health_endpoint"]),
                port=cast(int, config["port"]),
                timeout=cast(int, config.get("timeout", 10)),
                retries=cast(int, config.get("retries", 3)),
                dependencies=cast(list[str], config.get("dependencies", [])),
                critical=cast(bool, config.get("critical", True)),
                networks=cast(list[str], config.get("networks", None)),
                service_discovery=cast(
                    dict[str, str], config.get("service_discovery", None)
                ),
            )

        self.logger.info(
            "Loaded %d service configurations", len(self.services)
        )
        return self.services

    def get_service(self, name: str) -> ServiceConfig | None:
        """
        Get service configuration by name. Args: name: Service name Returns:
        Service configuration or None if not found
        """
        return self.services.get(name)

    def get_all_services(self) -> dict[str, ServiceConfig]:
        """
        Get all service configurations. Returns: Dictionary of all service
        configurations
        """
        return self.services.copy()

    def add_service(self, service: ServiceConfig) -> None:
        """
        Add or update a service configuration. Args: service: Service
        configuration to add
        """
        self.services[service.name] = service
        self.logger.info(
            "Added/updated service configuration: %s", service.name
        )

    def remove_service(self, name: str) -> bool:
        """
        Remove a service configuration. Args: name: Service name to remove
        Returns: True if service was removed, False if not found
        """
        if name in self.services:
            del self.services[name]
            self.logger.info("Removed service configuration: %s", name)
            return True
        return False

    def save_configuration(self, output_path: Path | None = None) -> None:
        """
        Save current configuration to file. Args: output_path: Optional custom
        output path
        """
        save_path = output_path or self.config_path

        config_data = {}
        for name, service in self.services.items():
            config_data[name] = {
                "name": service.name,
                "container_name": service.container_name,
                "health_endpoint": service.health_endpoint,
                "port": service.port,
                "timeout": service.timeout,
                "retries": service.retries,
                "dependencies": service.dependencies,
                "critical": service.critical,
                "networks": service.networks,
                "service_discovery": service.service_discovery,
            }

        with open(save_path, "w") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

        self.logger.info("Configuration saved to %s", save_path)

    def _get_default_configuration(self) -> dict[str, dict[str, Any]]:
        """
        Get default service configuration. Returns: Dictionary with default
        service configurations
        """
        return {
            "streamlit-app": {
                "name": "streamlit-app",
                "container_name": "crackseg-streamlit-app",
                "health_endpoint": "http://streamlit:8501/_stcore/health",
                "port": 8501,
                "timeout": 15,
                "retries": 3,
                "dependencies": [],
                "critical": True,
                "networks": [
                    "crackseg-frontend-network",
                    "crackseg-management-network",
                ],
                "service_discovery": {
                    "frontend": "streamlit:8501",
                    "management": "streamlit-mgmt:8501",
                },
            },
            "selenium-hub": {
                "name": "selenium-hub",
                "container_name": "crackseg-selenium-hub",
                "health_endpoint": "http://hub:4444/wd/hub/status",
                "port": 4444,
                "timeout": 10,
                "retries": 3,
                "dependencies": [],
                "critical": True,
                "networks": [
                    "crackseg-backend-network",
                    "crackseg-management-network",
                ],
                "service_discovery": {
                    "backend": "hub:4444",
                    "management": "hub-mgmt:4444",
                },
            },
            "chrome-node": {
                "name": "chrome-node",
                "container_name": "crackseg-chrome-node",
                "health_endpoint": "http://chrome:5555/wd/hub/status",
                "port": 5555,
                "timeout": 10,
                "retries": 3,
                "dependencies": ["selenium-hub"],
                "critical": True,
                "networks": [
                    "crackseg-backend-network",
                    "crackseg-management-network",
                ],
                "service_discovery": {
                    "backend": "chrome:5555",
                    "management": "chrome-mgmt:5555",
                },
            },
            "firefox-node": {
                "name": "firefox-node",
                "container_name": "crackseg-firefox-node",
                "health_endpoint": "http://firefox:5555/wd/hub/status",
                "port": 5555,
                "timeout": 10,
                "retries": 3,
                "dependencies": ["selenium-hub"],
                "critical": True,
                "networks": [
                    "crackseg-backend-network",
                    "crackseg-management-network",
                ],
                "service_discovery": {
                    "backend": "firefox:5555",
                    "management": "firefox-mgmt:5555",
                },
            },
            "test-runner": {
                "name": "test-runner",
                "container_name": "crackseg-test-runner",
                "health_endpoint": "http://test-runner:8080/health",
                "port": 8080,
                "timeout": 5,
                "retries": 2,
                "dependencies": [
                    "streamlit-app",
                    "selenium-hub",
                    "chrome-node",
                ],
                "critical": False,
                "networks": [
                    "crackseg-frontend-network",
                    "crackseg-backend-network",
                    "crackseg-management-network",
                ],
                "service_discovery": {
                    "frontend": "test-runner-frontend:8080",
                    "backend": "test-runner:8080",
                    "management": "test-runner-mgmt:8080",
                },
            },
            "grid-console": {
                "name": "grid-console",
                "container_name": "crackseg-grid-console",
                "health_endpoint": "http://console:4444/grid/api/hub/status",
                "port": 4444,
                "timeout": 10,
                "retries": 3,
                "dependencies": ["selenium-hub"],
                "critical": False,
                "networks": ["crackseg-management-network"],
                "service_discovery": {"management": "console:4444"},
            },
        }
