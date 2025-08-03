"""HTTP endpoint health verification."""

import json
import logging
from typing import Any

import requests

from ..models import HealthStatus, ServiceConfig


class EndpointChecker:
    """HTTP endpoint health verification."""

    def __init__(self) -> None:
        """Initialize endpoint checker."""
        self.logger = logging.getLogger("endpoint_checker")

    async def check_service_endpoint(
        self, service: ServiceConfig
    ) -> dict[str, Any]:
        """
        Check service-specific endpoint health. Args: service: Service
        configuration Returns: Dictionary with endpoint check results
        """
        try:
            response = requests.get(
                service.health_endpoint,
                timeout=service.timeout,
                verify=False,  # For internal Docker network
            )

            if response.status_code == 200:
                try:
                    response_data = response.json()
                    return {
                        "status": HealthStatus.HEALTHY,
                        "http_status": response.status_code,
                        "response_data": response_data,
                        "headers": dict(response.headers),
                    }
                except json.JSONDecodeError:
                    # Response is not JSON, but 200 OK is good enough
                    return {
                        "status": HealthStatus.HEALTHY,
                        "http_status": response.status_code,
                        "content_type": response.headers.get(
                            "content-type", "unknown"
                        ),
                    }
            else:
                return {
                    "status": HealthStatus.UNHEALTHY,
                    "http_status": response.status_code,
                    "error": f"HTTP {response.status_code}",
                }

        except requests.exceptions.Timeout:
            return {
                "status": HealthStatus.UNHEALTHY,
                "error": "Request timeout",
            }
        except requests.exceptions.ConnectionError:
            return {
                "status": HealthStatus.UNHEALTHY,
                "error": "Connection refused",
            }
        except Exception as e:
            return {"status": HealthStatus.UNHEALTHY, "error": str(e)}

    def validate_endpoint_config(self, service: ServiceConfig) -> bool:
        """
        Validate endpoint configuration. Args: service: Service configuration
        to validate Returns: True if configuration is valid
        """
        if not service.health_endpoint:
            self.logger.error(
                "Missing health endpoint for service %s", service.name
            )
            return False

        if not service.health_endpoint.startswith(("http://", "https://")):
            self.logger.error(
                "Invalid health endpoint URL for service %s: %s",
                service.name,
                service.health_endpoint,
            )
            return False

        if service.timeout <= 0:
            self.logger.error(
                "Invalid timeout for service %s: %s",
                service.name,
                service.timeout,
            )
            return False

        return True
