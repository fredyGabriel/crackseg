"""Continuous health monitoring."""

import asyncio
import logging

from .health_orchestrator import HealthOrchestrator
from .service_registry import ServiceRegistry


class ContinuousMonitor:
    """Continuous health monitoring service."""

    def __init__(
        self,
        orchestrator: HealthOrchestrator,
        service_registry: ServiceRegistry,
    ) -> None:
        """Initialize continuous monitor.

        Args:
            orchestrator: Health check orchestrator
            service_registry: Service configuration registry
        """
        self.logger = logging.getLogger("continuous_monitor")
        self.orchestrator = orchestrator
        self.service_registry = service_registry
        self.monitoring_active = False

    async def start_monitoring(self, interval: int = 30) -> None:
        """
        Start continuous health monitoring. Args: interval: Check interval in
        seconds
        """
        self.monitoring_active = True
        self.logger.info(
            "Starting continuous health monitoring (interval: %ds)", interval
        )

        while self.monitoring_active:
            try:
                services = self.service_registry.get_all_services()
                await self.orchestrator.check_all_services(services)
                await asyncio.sleep(interval)
            except Exception as e:
                self.logger.error("Error during monitoring cycle: %s", e)
                await asyncio.sleep(5)  # Short delay before retry

    def stop_monitoring(self) -> None:
        """Stop continuous health monitoring."""
        self.monitoring_active = False
        self.logger.info("Health monitoring stopped")
