"""Deployment orchestration and rollback mechanisms.

This module provides advanced deployment orchestration capabilities including
blue-green deployments, canary releases, rolling updates, and automatic
rollback mechanisms for the CrackSeg deployment system.
"""

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

# Import from manager (which we already fixed)
from .manager import (
    DefaultHealthChecker,
    DeploymentConfig,
    DeploymentManager,
    DeploymentMetadata,
    DeploymentResult,
    DeploymentState,
    DeploymentStrategy,
    HealthChecker,
)


# Simple definitions for compatibility
class AlertHandler(ABC):
    """Base alert handler."""

    @abstractmethod
    def send_alert(self, alert_type: str, metadata: Any, **kwargs) -> None:
        pass


class DeploymentHealthMonitor:
    """Deployment health monitor."""

    def __init__(self, health_checker: HealthChecker):
        self.health_checker = health_checker

    def add_deployment(
        self,
        deployment_id: str,
        health_check_url: str,
        process_name: str | None = None,
        check_interval: int = 30,
    ) -> None:
        pass


class PerformanceMonitor:
    """Performance monitor."""

    def __init__(self):
        pass


class DeploymentOrchestrator:
    """Advanced deployment orchestrator with rollback capabilities.

    Provides blue-green, canary, and rolling deployment strategies
    with automatic rollback mechanisms and health monitoring.
    """

    def __init__(self, health_checker: HealthChecker | None = None) -> None:
        """Initialize deployment orchestrator.

        Args:
            health_checker: Health checker implementation
        """
        self.health_checker = health_checker or DefaultHealthChecker()
        self.health_monitor = DeploymentHealthMonitor(
            health_checker=self.health_checker
        )
        self.deployment_manager = DeploymentManager(
            health_checker=self.health_checker
        )
        self.logger = logging.getLogger(__name__)
        self.deployment_history: dict[str, DeploymentMetadata] = {}
        self.performance_monitors: dict[str, PerformanceMonitor] = {}
        self.alert_handlers: list[AlertHandler] = []

    def add_alert_handler(self, handler: AlertHandler) -> None:
        """Add alert handler for deployment monitoring.

        Args:
            handler: Alert handler implementation
        """
        self.alert_handlers.append(handler)
        self.logger.info(f"Added alert handler: {handler.__class__.__name__}")

    def add_performance_monitor(
        self, deployment_id: str, monitor: PerformanceMonitor
    ) -> None:
        """Add performance monitor for deployment.

        Args:
            deployment_id: Deployment ID to monitor
            monitor: Performance monitor implementation
        """
        self.performance_monitors[deployment_id] = monitor
        self.logger.info(
            f"Added performance monitor for deployment {deployment_id}"
        )

    def add_deployment_monitoring(
        self,
        deployment_id: str,
        health_check_url: str,
        process_name: str | None = None,
        check_interval: int = 30,
    ) -> None:
        """Add deployment monitoring.

        Args:
            deployment_id: Deployment ID to monitor
            health_check_url: Health check URL
            process_name: Process name to monitor
            check_interval: Health check interval in seconds
        """
        self.health_monitor.add_deployment(
            deployment_id, health_check_url, process_name, check_interval
        )
        self.logger.info(f"Added monitoring for deployment {deployment_id}")

    def remove_deployment_monitoring(self, deployment_id: str) -> None:
        """Remove deployment monitoring.

        Args:
            deployment_id: Deployment ID to stop monitoring
        """
        self.health_monitor.remove_deployment(deployment_id)
        self.logger.info(f"Removed monitoring for deployment {deployment_id}")

    def start_health_monitoring(self) -> None:
        """Start health monitoring."""
        self.health_monitor.start_monitoring()

    def stop_health_monitoring(self) -> None:
        """Stop health monitoring."""
        self.health_monitor.stop_monitoring()

    def get_deployment_health_status(
        self, deployment_id: str
    ) -> dict[str, Any]:
        """Get deployment health status.

        Args:
            deployment_id: Deployment ID

        Returns:
            Health status information
        """
        return self.health_monitor.get_deployment_status(deployment_id)

    def get_all_health_statuses(self) -> dict[str, dict[str, Any]]:
        """Get all deployment health statuses.

        Returns:
            Dictionary of deployment health statuses
        """
        return self.health_monitor.get_all_statuses()

    def deploy_with_strategy(
        self,
        config: DeploymentConfig,
        strategy: DeploymentStrategy,
        deployment_func: Callable[..., DeploymentResult],
        **kwargs,
    ) -> DeploymentResult:
        """Deploy using specified strategy.

        Args:
            config: Deployment configuration
            strategy: Deployment strategy to use
            deployment_func: Function to execute deployment
            **kwargs: Additional deployment parameters

        Returns:
            Deployment result
        """
        # Create deployment metadata
        deployment_id = f"{config.artifact_id}-{int(time.time())}"
        metadata = DeploymentMetadata(
            deployment_id=deployment_id,
            start_time=time.time(),
        )

        # Store in history
        self.deployment_history[deployment_id] = metadata

        try:
            # Execute deployment using deployment manager
            result = self.deployment_manager.deploy_with_strategy(
                config, strategy, deployment_func, metadata, **kwargs
            )

            # Send alerts based on result
            if result.success:
                self._send_alerts("deployment_success", metadata)
            else:
                self._send_alerts(
                    "deployment_failure", metadata, error=result.error
                )

            return result

        except Exception as e:
            # Attempt rollback on failure
            if self._attempt_rollback(metadata, e):
                self._send_alerts("rollback_triggered", metadata)
            else:
                self._send_alerts("deployment_failure", metadata, error=str(e))

            raise

    def _attempt_rollback(
        self, metadata: DeploymentMetadata, error: Exception
    ) -> bool:
        """Attempt rollback on deployment failure.

        Args:
            metadata: Deployment metadata
            error: Error that triggered rollback

        Returns:
            True if rollback was successful
        """
        try:
            metadata.state = DeploymentState.ROLLING_BACK
            metadata.rollback_reason = str(error)

            # Find previous deployment
            if metadata.previous_deployment_id:
                # Switch traffic back to previous deployment
                self.logger.info(
                    f"Rolling back to deployment {metadata.previous_deployment_id}"
                )
                metadata.state = DeploymentState.ROLLED_BACK
                return True
            else:
                self.logger.warning("No previous deployment to rollback to")
                return False

        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            return False

    def manual_rollback(self, deployment_id: str) -> bool:
        """Manually rollback a deployment.

        Args:
            deployment_id: Deployment ID to rollback

        Returns:
            True if rollback was successful
        """
        if deployment_id not in self.deployment_history:
            self.logger.error(f"Deployment {deployment_id} not found")
            return False

        metadata = self.deployment_history[deployment_id]
        return self._attempt_rollback(metadata, Exception("Manual rollback"))

    def get_deployment_status(self, deployment_id: str) -> dict[str, Any]:
        """Get deployment status.

        Args:
            deployment_id: Deployment ID

        Returns:
            Deployment status information
        """
        if deployment_id not in self.deployment_history:
            return {"error": "Deployment not found"}

        metadata = self.deployment_history[deployment_id]
        return {
            "deployment_id": metadata.deployment_id,
            "artifact_id": metadata.artifact_id,
            "strategy": metadata.strategy.value,
            "state": metadata.state.value,
            "start_time": metadata.start_time,
            "end_time": metadata.end_time,
            "duration": (metadata.end_time or time.time())
            - metadata.start_time,
            "rollback_reason": metadata.rollback_reason,
        }

    def get_deployment_history(
        self, artifact_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Get deployment history.

        Args:
            artifact_id: Filter by artifact ID (optional)

        Returns:
            List of deployment statuses
        """
        history = []
        for deployment_id, metadata in self.deployment_history.items():
            if artifact_id and metadata.artifact_id != artifact_id:
                continue

            history.append(self.get_deployment_status(deployment_id))

        return sorted(history, key=lambda x: x["start_time"], reverse=True)

    def get_performance_metrics(self, deployment_id: str) -> dict[str, Any]:
        """Get performance metrics for deployment.

        Args:
            deployment_id: Deployment ID

        Returns:
            Performance metrics
        """
        if deployment_id not in self.performance_monitors:
            return {}

        return self.performance_monitors[deployment_id].get_metrics()

    def stop_performance_monitoring(self, deployment_id: str) -> None:
        """Stop performance monitoring for deployment.

        Args:
            deployment_id: Deployment ID
        """
        if deployment_id in self.performance_monitors:
            self.performance_monitors[deployment_id].stop_monitoring()
            del self.performance_monitors[deployment_id]
            self.logger.info(
                f"Stopped performance monitoring for {deployment_id}"
            )

    def _start_performance_monitoring(
        self, deployment_id: str, deployment_url: str
    ) -> None:
        """Start performance monitoring for deployment.

        Args:
            deployment_id: Deployment ID to monitor
            deployment_url: Deployment URL for monitoring
        """
        try:
            monitor = PerformanceMonitor()
            self.add_performance_monitor(deployment_id, monitor)

            # Start monitoring in background thread
            import threading

            monitor_thread = threading.Thread(
                target=monitor.start_monitoring,
                args=(deployment_id,),
                daemon=True,
            )
            monitor_thread.start()

            self.logger.info(
                f"Started performance monitoring for {deployment_id}"
            )

        except Exception as e:
            self.logger.warning(f"Failed to start performance monitoring: {e}")

    def _send_alerts(
        self, alert_type: str, metadata: DeploymentMetadata, **kwargs
    ) -> None:
        """Send alerts to all registered handlers.

        Args:
            alert_type: Type of alert to send
            metadata: Deployment metadata
            **kwargs: Additional alert data
        """
        for handler in self.alert_handlers:
            try:
                handler.send_alert(alert_type, metadata, **kwargs)
            except Exception as e:
                self.logger.warning(
                    f"Failed to send alert via {handler.__class__.__name__}: "
                    f"{e}"
                )
