"""Health monitoring integration for deployment system.

This module integrates the existing health check system with the deployment
orchestration to provide comprehensive monitoring capabilities.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import psutil
import requests

logger = logging.getLogger(__name__)


@dataclass
class HealthCheckResult:
    """Result of a health check operation."""

    success: bool
    status: str  # "healthy", "degraded", "unhealthy"
    response_time_ms: float
    timestamp: float
    details: dict[str, Any] | None = None
    error_message: str | None = None


@dataclass
class ResourceMetrics:
    """System resource metrics."""

    cpu_usage_percent: float
    memory_usage_mb: float
    disk_usage_percent: float
    network_io_mbps: float
    timestamp: float


class HealthChecker(Protocol):
    """Protocol for health checker implementations."""

    def check_health(self, url: str, timeout: int = 10) -> HealthCheckResult:
        """Check health of a service endpoint."""

    def wait_for_healthy(
        self, url: str, max_wait: int = 300, interval: int = 5
    ) -> bool:
        """Wait for service to become healthy."""


class ResourceMonitor(Protocol):
    """Protocol for resource monitoring implementations."""

    def get_system_metrics(self) -> ResourceMetrics:
        """Get current system resource metrics."""

    def get_process_metrics(self, process_name: str) -> ResourceMetrics:
        """Get metrics for a specific process."""


class DefaultHealthChecker:
    """Default HTTP-based health checker implementation."""

    def __init__(self) -> None:
        """Initialize health checker."""
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "CrackSeg-HealthChecker/1.0"}
        )

    def check_health(self, url: str, timeout: int = 10) -> HealthCheckResult:
        """Check health of a service endpoint.

        Args:
            url: Health check endpoint URL
            timeout: Request timeout in seconds

        Returns:
            Health check result
        """
        start_time = time.time()
        timestamp = time.time()

        try:
            response = self.session.get(
                url, timeout=timeout, allow_redirects=False
            )
            response_time_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                return HealthCheckResult(
                    success=True,
                    status="healthy",
                    response_time_ms=response_time_ms,
                    timestamp=timestamp,
                    details={
                        "status_code": response.status_code,
                        "content_length": len(response.content),
                    },
                )
            else:
                return HealthCheckResult(
                    success=False,
                    status="unhealthy",
                    response_time_ms=response_time_ms,
                    timestamp=timestamp,
                    error_message=f"HTTP {response.status_code}",
                    details={"status_code": response.status_code},
                )

        except requests.exceptions.RequestException as e:
            response_time_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                success=False,
                status="unhealthy",
                response_time_ms=response_time_ms,
                timestamp=timestamp,
                error_message=str(e),
            )

    def wait_for_healthy(
        self, url: str, max_wait: int = 300, interval: int = 5
    ) -> bool:
        """Wait for service to become healthy.

        Args:
            url: Health check endpoint URL
            max_wait: Maximum wait time in seconds
            interval: Check interval in seconds

        Returns:
            True if service becomes healthy, False otherwise
        """
        start_time = time.time()
        logger.info(f"Waiting for {url} to become healthy (max {max_wait}s)")

        while time.time() - start_time < max_wait:
            result = self.check_health(url)
            if result.success:
                logger.info(f"Service {url} is healthy")
                return True

            logger.debug(
                f"Service {url} not healthy: {result.error_message}, "
                f"retrying in {interval}s"
            )
            time.sleep(interval)

        logger.error(
            f"Service {url} did not become healthy within {max_wait}s"
        )
        return False


class DefaultResourceMonitor:
    """Default system resource monitor implementation."""

    def get_system_metrics(self) -> ResourceMetrics:
        """Get current system resource metrics.

        Returns:
            System resource metrics
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            network = psutil.net_io_counters()

            # Calculate network IO in Mbps
            network_io_mbps = (
                (network.bytes_sent + network.bytes_recv) * 8 / 1_000_000
            )

            return ResourceMetrics(
                cpu_usage_percent=cpu_percent,
                memory_usage_mb=memory.used / 1024 / 1024,
                disk_usage_percent=disk.percent,
                network_io_mbps=network_io_mbps,
                timestamp=time.time(),
            )

        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return ResourceMetrics(
                cpu_usage_percent=0.0,
                memory_usage_mb=0.0,
                disk_usage_percent=0.0,
                network_io_mbps=0.0,
                timestamp=time.time(),
            )

    def get_process_metrics(self, process_name: str) -> ResourceMetrics:
        """Get metrics for a specific process.

        Args:
            process_name: Name of the process to monitor

        Returns:
            Process resource metrics
        """
        try:
            for proc in psutil.process_iter(
                ["pid", "name", "cpu_percent", "memory_info"]
            ):
                if (
                    proc.info["name"]
                    and process_name.lower() in proc.info["name"].lower()
                ):
                    memory_info = proc.info["memory_info"]
                    return ResourceMetrics(
                        cpu_usage_percent=proc.info["cpu_percent"] or 0.0,
                        memory_usage_mb=memory_info.rss / 1024 / 1024,
                        disk_usage_percent=0.0,  # Process-level disk usage not easily available  # noqa: E501
                        network_io_mbps=0.0,  # Process-level network IO not easily available  # noqa: E501
                        timestamp=time.time(),
                    )

            # Process not found
            return ResourceMetrics(
                cpu_usage_percent=0.0,
                memory_usage_mb=0.0,
                disk_usage_percent=0.0,
                network_io_mbps=0.0,
                timestamp=time.time(),
            )

        except Exception as e:
            logger.error(
                f"Error getting process metrics for {process_name}: {e}"
            )
            return ResourceMetrics(
                cpu_usage_percent=0.0,
                memory_usage_mb=0.0,
                disk_usage_percent=0.0,
                network_io_mbps=0.0,
                timestamp=time.time(),
            )


class DeploymentHealthMonitor:
    """Comprehensive health monitoring for deployments.

    Integrates health checks, resource monitoring, and alerting
    for deployed artifacts.
    """

    def __init__(
        self,
        health_checker: HealthChecker | None = None,
        resource_monitor: ResourceMonitor | None = None,
    ) -> None:
        """Initialize deployment health monitor.

        Args:
            health_checker: Health checker implementation
            resource_monitor: Resource monitor implementation
        """
        self.health_checker = health_checker or DefaultHealthChecker()
        self.resource_monitor = resource_monitor or DefaultResourceMonitor()
        self.logger = logging.getLogger(__name__)
        self.monitoring_active = False
        self.monitored_deployments: dict[str, dict[str, Any]] = {}

        # Alert thresholds
        self.alert_thresholds = {
            "response_time_ms": 1000.0,
            "cpu_usage_percent": 80.0,
            "memory_usage_mb": 1536.0,  # 1.5GB
            "error_rate_percent": 5.0,
        }

        logger.info("DeploymentHealthMonitor initialized")

    def add_deployment_monitoring(
        self,
        deployment_id: str,
        health_check_url: str,
        process_name: str | None = None,
        check_interval: int = 30,
    ) -> None:
        """Add deployment to monitoring.

        Args:
            deployment_id: Unique deployment identifier
            health_check_url: Health check endpoint URL
            process_name: Name of the process to monitor (optional)
            check_interval: Health check interval in seconds
        """
        self.monitored_deployments[deployment_id] = {
            "health_check_url": health_check_url,
            "process_name": process_name,
            "check_interval": check_interval,
            "last_check": 0.0,
            "health_history": [],
            "resource_history": [],
        }

        logger.info(f"Added deployment {deployment_id} to monitoring")

    def remove_deployment_monitoring(self, deployment_id: str) -> None:
        """Remove deployment from monitoring.

        Args:
            deployment_id: Deployment identifier to remove
        """
        if deployment_id in self.monitored_deployments:
            del self.monitored_deployments[deployment_id]
            logger.info(f"Removed deployment {deployment_id} from monitoring")

    def start_monitoring(self) -> None:
        """Start continuous monitoring."""
        self.monitoring_active = True
        logger.info("Started deployment health monitoring")

        # Start monitoring in background
        asyncio.create_task(self._monitoring_loop())

    def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        self.monitoring_active = False
        logger.info("Stopped deployment health monitoring")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                current_time = time.time()

                for (
                    deployment_id,
                    config,
                ) in self.monitored_deployments.items():
                    # Check if it's time for health check
                    if (
                        current_time - config["last_check"]
                        >= config["check_interval"]
                    ):
                        await self._check_deployment_health(
                            deployment_id, config
                        )
                        config["last_check"] = current_time

                await asyncio.sleep(10)  # Check every 10 seconds

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)

    async def _check_deployment_health(
        self, deployment_id: str, config: dict[str, Any]
    ) -> None:
        """Check health of a specific deployment.

        Args:
            deployment_id: Deployment identifier
            config: Deployment monitoring configuration
        """
        try:
            # Health check
            health_result = self.health_checker.check_health(
                config["health_check_url"]
            )
            config["health_history"].append(health_result)

            # Resource monitoring
            resource_metrics = self.resource_monitor.get_system_metrics()
            if config["process_name"]:
                process_metrics = self.resource_monitor.get_process_metrics(
                    config["process_name"]
                )
                resource_metrics = process_metrics

            config["resource_history"].append(resource_metrics)

            # Check for alerts
            self._check_alerts(deployment_id, health_result, resource_metrics)

            # Keep history manageable (last 100 entries)
            if len(config["health_history"]) > 100:
                config["health_history"] = config["health_history"][-100:]
            if len(config["resource_history"]) > 100:
                config["resource_history"] = config["resource_history"][-100:]

        except Exception as e:
            logger.error(f"Error checking health for {deployment_id}: {e}")

    def _check_alerts(
        self,
        deployment_id: str,
        health_result: HealthCheckResult,
        resource_metrics: ResourceMetrics,
    ) -> None:
        """Check for alert conditions.

        Args:
            deployment_id: Deployment identifier
            health_result: Health check result
            resource_metrics: Resource metrics
        """
        alerts = []

        # Health check alerts
        if not health_result.success:
            alerts.append(
                f"Health check failed: {health_result.error_message}"
            )

        if (
            health_result.response_time_ms
            > self.alert_thresholds["response_time_ms"]
        ):
            alerts.append(
                f"High response time: {health_result.response_time_ms:.1f}ms"
            )

        # Resource alerts
        if (
            resource_metrics.cpu_usage_percent
            > self.alert_thresholds["cpu_usage_percent"]
        ):
            alerts.append(
                f"High CPU usage: {resource_metrics.cpu_usage_percent:.1f}%"
            )

        if (
            resource_metrics.memory_usage_mb
            > self.alert_thresholds["memory_usage_mb"]
        ):
            alerts.append(
                f"High memory usage: {resource_metrics.memory_usage_mb:.1f}MB"
            )

        # Log alerts
        for alert in alerts:
            logger.warning(f"Alert for {deployment_id}: {alert}")

    def get_deployment_status(self, deployment_id: str) -> dict[str, Any]:
        """Get current status of a deployment.

        Args:
            deployment_id: Deployment identifier

        Returns:
            Deployment status information
        """
        if deployment_id not in self.monitored_deployments:
            return {"error": "Deployment not found"}

        config = self.monitored_deployments[deployment_id]
        health_history = config["health_history"]
        resource_history = config["resource_history"]

        if not health_history:
            return {
                "status": "unknown",
                "message": "No health checks performed",
            }

        latest_health = health_history[-1]
        latest_resource = resource_history[-1] if resource_history else None

        return {
            "deployment_id": deployment_id,
            "health_status": latest_health.status,
            "response_time_ms": latest_health.response_time_ms,
            "last_check": latest_health.timestamp,
            "resource_metrics": (
                latest_resource.__dict__ if latest_resource else None
            ),
            "health_history_count": len(health_history),
            "resource_history_count": len(resource_history),
        }

    def get_all_deployment_statuses(self) -> dict[str, dict[str, Any]]:
        """Get status of all monitored deployments.

        Returns:
            Dictionary mapping deployment IDs to their status
        """
        return {
            deployment_id: self.get_deployment_status(deployment_id)
            for deployment_id in self.monitored_deployments.keys()
        }

    def export_monitoring_data(self, output_path: Path) -> None:
        """Export monitoring data to file.

        Args:
            output_path: Path to export file
        """
        try:
            export_data = {
                "timestamp": time.time(),
                "monitored_deployments": self.monitored_deployments,
            }

            import json

            with open(output_path, "w") as f:
                json.dump(export_data, f, indent=2, default=str)

            logger.info(f"Exported monitoring data to {output_path}")

        except Exception as e:
            logger.error(f"Error exporting monitoring data: {e}")
