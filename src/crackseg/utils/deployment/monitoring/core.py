"""Core consolidated monitoring system."""

import logging
import threading
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import (
        AlertThresholds,
        DashboardConfig,
        HealthCheckConfig,
        MetricsConfig,
    )
    from .health import HealthChecker
    from .metrics import MetricsCollector
    from .performance import PerformanceMonitor
    from .resource import ResourceMonitor


class DeploymentMonitoringSystem:
    """Consolidated monitoring system for deployments."""

    def __init__(
        self,
        health_config: "HealthCheckConfig",
        metrics_config: "MetricsConfig",
        dashboard_config: "DashboardConfig",
        alert_thresholds: "AlertThresholds",
    ) -> None:
        """Initialize monitoring system.

        Args:
            health_config: Health check configuration
            metrics_config: Metrics configuration
            dashboard_config: Dashboard configuration
            alert_thresholds: Alert thresholds
        """
        self.health_config = health_config
        self.metrics_config = metrics_config
        self.dashboard_config = dashboard_config
        self.alert_thresholds = alert_thresholds

        # Initialize components
        self.health_checker = HealthChecker()
        self.resource_monitor = ResourceMonitor()
        self.metrics_collector = MetricsCollector(metrics_config)

        # Monitoring state
        self.monitoring = False
        self.monitor_thread = None
        self.deployments: dict[str, dict[str, Any]] = {}

        self.logger = logging.getLogger(__name__)

    def add_deployment(
        self,
        deployment_id: str,
        health_url: str,
        process_name: str | None = None,
    ) -> None:
        """Add deployment to monitoring.

        Args:
            deployment_id: Deployment ID
            health_url: Health check URL
            process_name: Process name for resource monitoring
        """
        self.deployments[deployment_id] = {
            "health_url": health_url,
            "process_name": process_name,
            "performance_monitor": PerformanceMonitor(
                health_url, self.metrics_config
            ),
            "added_at": time.time(),
        }

        self.logger.info(f"Added deployment {deployment_id} to monitoring")

    def remove_deployment(self, deployment_id: str) -> None:
        """Remove deployment from monitoring.

        Args:
            deployment_id: Deployment ID to remove
        """
        if deployment_id in self.deployments:
            deployment = self.deployments[deployment_id]
            deployment["performance_monitor"].stop_monitoring()
            del self.deployments[deployment_id]
            self.logger.info(
                f"Removed deployment {deployment_id} from monitoring"
            )

    def start_monitoring(self) -> None:
        """Start monitoring all deployments."""
        if self.monitoring:
            self.logger.warning("Monitoring already started")
            return

        self.monitoring = True
        self.logger.info("Starting deployment monitoring")

        def monitor_loop():
            while self.monitoring:
                try:
                    self._monitor_all_deployments()
                    time.sleep(self.metrics_config.collection_interval)
                except Exception as e:
                    self.logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(60)

        self.monitor_thread = threading.Thread(
            target=monitor_loop, daemon=True
        )
        self.monitor_thread.start()

    def stop_monitoring(self) -> None:
        """Stop monitoring all deployments."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

        # Stop all performance monitors
        for deployment in self.deployments.values():
            deployment["performance_monitor"].stop_monitoring()

        self.logger.info("Stopped deployment monitoring")

    def _monitor_all_deployments(self) -> None:
        """Monitor all registered deployments."""
        for deployment_id, deployment_config in self.deployments.items():
            try:
                self._monitor_deployment(deployment_id, deployment_config)
            except Exception as e:
                self.logger.error(
                    f"Error monitoring deployment {deployment_id}: {e}"
                )

    def _monitor_deployment(
        self, deployment_id: str, deployment_config: dict[str, Any]
    ) -> None:
        """Monitor a single deployment.

        Args:
            deployment_id: Deployment ID
            deployment_config: Deployment configuration
        """
        # Health check
        health_config = HealthCheckConfig(
            url=deployment_config["health_url"],
            timeout=self.health_config.timeout,
            interval=self.health_config.interval,
        )
        health_result = self.health_checker.check_health(health_config)

        # Resource monitoring
        if deployment_config["process_name"]:
            resource_metrics = self.resource_monitor.get_process_metrics(
                deployment_config["process_name"]
            )
        else:
            resource_metrics = self.resource_monitor.get_system_metrics()

        # Performance monitoring
        performance_monitor = deployment_config["performance_monitor"]
        performance_metrics = performance_monitor.get_metrics()

        # Collect and aggregate metrics
        monitoring_result = self.metrics_collector.collect_metrics(
            health_result, resource_metrics, performance_metrics
        )

        # Store result
        deployment_config["last_result"] = monitoring_result
        deployment_config["last_check"] = time.time()

        # Log alerts
        if monitoring_result.alerts:
            for alert in monitoring_result.alerts:
                self.logger.warning(f"Deployment {deployment_id}: {alert}")

    def get_deployment_status(self, deployment_id: str) -> dict[str, Any]:
        """Get status of a specific deployment.

        Args:
            deployment_id: Deployment ID

        Returns:
            Deployment status
        """
        if deployment_id not in self.deployments:
            return {"error": "Deployment not found"}

        deployment = self.deployments[deployment_id]
        last_result = deployment.get("last_result")

        if not last_result:
            return {"status": "unknown", "message": "No monitoring data"}

        return {
            "deployment_id": deployment_id,
            "status": last_result.health_status,
            "success": last_result.success,
            "timestamp": last_result.timestamp,
            "metrics": last_result.metrics,
            "alerts": last_result.alerts,
            "error_message": last_result.error_message,
        }

    def get_all_deployment_statuses(self) -> dict[str, dict[str, Any]]:
        """Get status of all deployments.

        Returns:
            Dictionary of deployment statuses
        """
        return {
            deployment_id: self.get_deployment_status(deployment_id)
            for deployment_id in self.deployments
        }

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get summary of all collected metrics.

        Returns:
            Metrics summary
        """
        return self.metrics_collector.get_metrics_summary()

    def export_monitoring_data(self, format: str = "json") -> str:
        """Export monitoring data.

        Args:
            format: Export format

        Returns:
            Exported data
        """
        return self.metrics_collector.export_metrics(format)
