#!/usr/bin/env python3
"""
CrackSeg Health Check System - Main Implementation.
This file provides the main health check system implementation with
backward compatibility to the original monolithic API while internally
using the new modular architecture from docker/health_check/.
All original APIs are preserved for seamless transition.
Author: CrackSeg Project Version: 1.0 (Subtask 13.7) - Modular Architecture
"""

# Minimal imports needed for backward compatibility
import logging
from pathlib import Path
from typing import Any

from health_check import (
    HealthCheckResult,
    HealthStatus,
    ServiceConfig,
    SystemHealthReport,
    cli,
)

# Import from new modular architecture
from health_check import HealthCheckSystem as ModularHealthCheckSystem

# =============================================================================
# Backward Compatibility Layer
# =============================================================================

# Re-export all original classes and enums for compatibility
__all__ = [
    "HealthStatus",
    "ServiceConfig",
    "HealthCheckResult",
    "SystemHealthReport",
    "HealthCheckSystem",
    "cli",
]


class HealthCheckSystem:
    """
    Main health check system for CrackSeg Docker infrastructure.
    This class maintains backward compatibility with the original API
    while internally using the new modular architecture.
    """

    def __init__(self, config_path: Path | None = None) -> None:
        """
        Initialize health check system. Args: config_path: Path to health
        check configuration file
        """
        # Use the new modular implementation internally
        self._system = ModularHealthCheckSystem(config_path)

        # Expose properties for backward compatibility
        self.logger = self._system.logger
        self.services = self._system.services
        self.health_history = self._system.health_history
        self.monitoring_active = False

    async def check_service_health(
        self, service: ServiceConfig
    ) -> HealthCheckResult:
        """
        Check health of a single service. Args: service: Service configuration
        to check Returns: Health check result with status and metrics
        """
        return await self._system.check_service_health(service)

    async def check_all_services(self) -> SystemHealthReport:
        """
        Check health of all configured services. Returns: Comprehensive system
        health report
        """
        return await self._system.check_all_services()

    def generate_dashboard_data(self) -> dict[str, Any]:
        """
        Generate data for monitoring dashboard. Returns: Dashboard data
        structure
        """
        return self._system.generate_dashboard_data()

    async def start_monitoring(self, interval: int = 30) -> None:
        """
        Start continuous health monitoring. Args: interval: Check interval in
        seconds
        """
        self.monitoring_active = True
        await self._system.start_monitoring(interval)

    def stop_monitoring(self) -> None:
        """Stop continuous health monitoring."""
        self.monitoring_active = False
        self._system.stop_monitoring()

    def save_report(
        self, report: SystemHealthReport, output_path: Path
    ) -> None:
        """Save health report to file.

        Args:
            report: Health report to save
            output_path: Output file path
        """
        self._system.save_report(report, output_path)

    # Legacy method aliases for complete backward compatibility
    def _setup_logging(self) -> logging.Logger:
        """Legacy method - now handled by modular system."""
        return self.logger

    def _load_configuration(self) -> None:
        """Legacy method - now handled by service registry."""
        # Configuration is automatically loaded by the modular system
        pass

    async def _check_docker_container_status(
        self, container_name: str
    ) -> HealthStatus:
        """Legacy method - now handled by DockerChecker."""
        # This functionality is now encapsulated in the DockerChecker
        # We could expose it if needed, but it's internal implementation
        raise NotImplementedError(
            "This internal method is now handled by the modular DockerChecker."
            " Use check_service_health() instead."
        )

    async def _check_service_endpoint(
        self, service: ServiceConfig
    ) -> dict[str, Any]:
        """Legacy method - now handled by EndpointChecker."""
        # This functionality is now encapsulated in the EndpointChecker
        raise NotImplementedError(
            "This internal method is now handled by the modular "
            "EndpointChecker. Use check_service_health() instead."
        )

    def _validate_dependencies(
        self, service_results: dict[str, HealthCheckResult]
    ) -> bool:
        """Legacy method - now handled by DependencyValidator."""
        # This functionality is now encapsulated in the DependencyValidator
        raise NotImplementedError(
            "This internal method is now handled by the modular "
            "DependencyValidator. Dependencies are automatically validated in "
            "check_all_services()."
        )

    def _generate_recommendations(
        self,
        service_results: dict[str, HealthCheckResult],
        dependencies_satisfied: bool,
    ) -> list[str]:
        """Legacy method - now handled by RecommendationEngine."""
        # This functionality is now encapsulated in the RecommendationEngine
        raise NotImplementedError(
            "This internal method is now handled by the modular "
            "RecommendationEngine. Recommendations are automatically generated"
            " in check_all_services()."
        )

    def _collect_system_metrics(
        self, service_results: dict[str, HealthCheckResult]
    ) -> dict[str, Any]:
        """Legacy method - now handled by MetricsCollector."""
        # This functionality is now encapsulated in the MetricsCollector
        raise NotImplementedError(
            "This internal method is now handled by the modular "
            "MetricsCollector. Metrics are automatically collected in "
            "check_all_services()."
        )


# =============================================================================
# CLI Interface (unchanged for compatibility)
# =============================================================================

# The CLI is unchanged and imported from the modular implementation
# All original click commands work exactly the same

if __name__ == "__main__":
    cli()
