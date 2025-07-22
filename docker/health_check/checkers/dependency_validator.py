"""Service dependency validation."""

import logging
from typing import Any

from ..models import HealthCheckResult, HealthStatus, ServiceConfig


class DependencyValidator:
    """Service dependency validation and analysis."""

    def __init__(self) -> None:
        """Initialize dependency validator."""
        self.logger = logging.getLogger("dependency_validator")

    def validate_dependencies(
        self,
        service_results: dict[str, HealthCheckResult],
        services: dict[str, ServiceConfig],
    ) -> bool:
        """
        Validate service dependencies are satisfied. Args: service_results:
        Results from health checks services: Service configurations Returns:
        True if all dependencies are satisfied
        """
        for service_name, service_config in services.items():
            service_result = service_results.get(service_name)

            if (
                not service_result
                or service_result.status != HealthStatus.HEALTHY
            ):
                continue

            # Check if all dependencies are healthy
            for dependency in service_config.dependencies or []:
                dep_result = service_results.get(dependency)
                if not dep_result or dep_result.status != HealthStatus.HEALTHY:
                    self.logger.warning(
                        "Service %s depends on unhealthy service %s",
                        service_name,
                        dependency,
                    )
                    return False

        return True

    def analyze_dependency_chain(
        self, services: dict[str, ServiceConfig]
    ) -> dict[str, Any]:
        """
        Analyze dependency chains for potential issues. Args: services:
        Service configurations Returns: Analysis results with potential issues
        """
        analysis = {
            "circular_dependencies": [],
            "orphaned_services": [],
            "critical_path": [],
            "dependency_depth": {},
        }

        # Check for circular dependencies
        for service_name in services:
            if self._has_circular_dependency(service_name, services):
                analysis["circular_dependencies"].append(service_name)

        # Find orphaned services (no dependents)
        all_dependencies = set()
        for service in services.values():
            all_dependencies.update(service.dependencies or [])

        for service_name in services:
            if service_name not in all_dependencies:
                analysis["orphaned_services"].append(service_name)

        # Calculate dependency depth
        for service_name in services:
            analysis["dependency_depth"][service_name] = self._calculate_depth(
                service_name, services
            )

        # Identify critical path (longest dependency chain)
        if analysis["dependency_depth"]:
            max_depth = max(analysis["dependency_depth"].values())
            analysis["critical_path"] = [
                name
                for name, depth in analysis["dependency_depth"].items()
                if depth == max_depth
            ]

        return analysis

    def _has_circular_dependency(
        self,
        service_name: str,
        services: dict[str, ServiceConfig],
        visited: set[str] | None = None,
    ) -> bool:
        """
        Check if a service has circular dependencies. Args: service_name:
        Service to check services: All service configurations visited: Set of
        already visited services Returns: True if circular dependency detected
        """
        if visited is None:
            visited = set()

        if service_name in visited:
            return True

        visited.add(service_name)

        service = services.get(service_name)
        if not service or not service.dependencies:
            visited.remove(service_name)
            return False

        for dependency in service.dependencies:
            if self._has_circular_dependency(dependency, services, visited):
                return True

        visited.remove(service_name)
        return False

    def _calculate_depth(
        self,
        service_name: str,
        services: dict[str, ServiceConfig],
        visited: set[str] | None = None,
    ) -> int:
        """
        Calculate dependency depth for a service. Args: service_name: Service
        to analyze services: All service configurations visited: Set of
        visited services (circular detection) Returns: Maximum dependency
        depth
        """
        if visited is None:
            visited = set()

        if service_name in visited:
            return 0  # Circular dependency, return 0

        visited.add(service_name)

        service = services.get(service_name)
        if not service or not service.dependencies:
            visited.remove(service_name)
            return 0

        max_depth = 0
        for dependency in service.dependencies:
            depth = self._calculate_depth(dependency, services, visited)
            max_depth = max(max_depth, depth + 1)

        visited.remove(service_name)
        return max_depth
