"""
CrackSeg Health Check System - Modular Implementation. Comprehensive
health check system for Docker services combining native Docker health
checks with detailed custom monitoring and observability. This module
provides: - Service health monitoring for all Docker services -
Dependency validation between services - Custom metrics collection and
reporting - Dashboard and logging capabilities - Automatic restart
mechanisms coordination Author: CrackSeg Project Version: 1.0 (Subtask
13.7) - Refactored Modular Architecture
"""

import logging
from pathlib import Path
from typing import Any

from .analytics import (
    DashboardGenerator,
    MetricsCollector,
    RecommendationEngine,
)
from .cli import cli
from .models import (
    HealthCheckResult,
    HealthStatus,
    ServiceConfig,
    SystemHealthReport,
)
from .orchestration import (
    ContinuousMonitor,
    HealthOrchestrator,
    ServiceRegistry,
)
from .persistence import ReportSaver


class HealthCheckSystem:
    """
    Comprehensive health check system for CrackSeg Docker infrastructure.
    This is the main facade class that provides backward compatibility
    with the original monolithic implementation while using the new
    modular architecture.
    """

    def __init__(self, config_path: Path | None = None) -> None:
        """
        Initialize health check system. Args: config_path: Path to health
        check configuration file
        """
        self.logger = self._setup_logging()

        # Initialize modular components
        self.service_registry = ServiceRegistry(config_path)
        self.orchestrator = HealthOrchestrator()
        self.report_saver = ReportSaver()
        self.monitor: ContinuousMonitor | None = None

        # Load configuration
        self.services = self.service_registry.load_configuration()

        self.logger.info(
            "HealthCheckSystem initialized with %d services",
            len(self.services),
        )

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger("health_check_system")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    async def check_service_health(
        self, service: ServiceConfig
    ) -> HealthCheckResult:
        """
        Check health of a single service. Args: service: Service configuration
        to check Returns: Health check result with status and metrics
        """
        return await self.orchestrator.check_service_health(service)

    async def check_all_services(self) -> SystemHealthReport:
        """
        Check health of all configured services. Returns: Comprehensive system
        health report
        """
        return await self.orchestrator.check_all_services(self.services)

    def generate_dashboard_data(self) -> dict[str, Any]:
        """
        Generate data for monitoring dashboard. Returns: Dashboard data
        structure
        """
        return self.orchestrator.generate_dashboard_data()

    async def start_monitoring(self, interval: int = 30) -> None:
        """
        Start continuous health monitoring. Args: interval: Check interval in
        seconds
        """
        if not self.monitor:
            self.monitor = ContinuousMonitor(
                self.orchestrator, self.service_registry
            )
        await self.monitor.start_monitoring(interval)

    def stop_monitoring(self) -> None:
        """Stop continuous health monitoring."""
        if self.monitor:
            self.monitor.stop_monitoring()

    def save_report(
        self, report: SystemHealthReport, output_path: Path
    ) -> None:
        """Save health report to file.

        Args:
            report: Health report to save
            output_path: Output file path
        """
        self.report_saver.save_report(report, output_path)

    @property
    def health_history(self) -> list[SystemHealthReport]:
        """Get health check history."""
        return self.orchestrator.get_health_history()


# Backward compatibility exports
__all__ = [
    # Main system class
    "HealthCheckSystem",
    # Data models
    "HealthStatus",
    "ServiceConfig",
    "HealthCheckResult",
    "SystemHealthReport",
    # CLI interface
    "cli",
    # Modular components (for advanced usage)
    "HealthOrchestrator",
    "ServiceRegistry",
    "ContinuousMonitor",
    "MetricsCollector",
    "RecommendationEngine",
    "DashboardGenerator",
    "ReportSaver",
]
