"""Deployment orchestration and rollback mechanisms.

This module provides advanced deployment orchestration capabilities including
blue-green deployments, canary releases, rolling updates, and automatic
rollback mechanisms for the CrackSeg deployment system.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .config import DeploymentConfig, DeploymentResult
from .health_monitoring import (
    DefaultHealthChecker,
    DeploymentHealthMonitor,
    HealthChecker,
)


class DeploymentStrategy(Enum):
    """Deployment strategies for different environments."""

    BLUE_GREEN = "blue-green"
    CANARY = "canary"
    ROLLING = "rolling"
    RECREATE = "recreate"


class DeploymentState(Enum):
    """Deployment states for tracking progress."""

    PENDING = "pending"
    IN_PROGRESS = "in-progress"
    HEALTH_CHECKING = "health-checking"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLING_BACK = "rolling-back"
    ROLLED_BACK = "rolled-back"


@dataclass
class DeploymentMetadata:
    """Metadata for tracking deployment information."""

    deployment_id: str
    artifact_id: str
    strategy: DeploymentStrategy
    state: DeploymentState
    start_time: float
    end_time: float | None = None
    previous_deployment_id: str | None = None
    rollback_reason: str | None = None
    health_check_url: str | None = None
    metrics_url: str | None = None


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
        self.logger = logging.getLogger(__name__)
        self.deployment_history: dict[str, DeploymentMetadata] = {}
        self.performance_monitors: dict[str, PerformanceMonitor] = {}
        self.alert_handlers: list[AlertHandler] = []

    def add_alert_handler(self, handler: "AlertHandler") -> None:
        """Add alert handler for deployment monitoring.

        Args:
            handler: Alert handler implementation
        """
        self.alert_handlers.append(handler)
        self.logger.info(f"Added alert handler: {handler.__class__.__name__}")

    def add_performance_monitor(
        self, deployment_id: str, monitor: "PerformanceMonitor"
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
        """Add deployment to health monitoring.

        Args:
            deployment_id: Deployment identifier
            health_check_url: Health check endpoint URL
            process_name: Name of the process to monitor (optional)
            check_interval: Health check interval in seconds
        """
        self.health_monitor.add_deployment_monitoring(
            deployment_id, health_check_url, process_name, check_interval
        )

    def remove_deployment_monitoring(self, deployment_id: str) -> None:
        """Remove deployment from health monitoring.

        Args:
            deployment_id: Deployment identifier to remove
        """
        self.health_monitor.remove_deployment_monitoring(deployment_id)

    def start_health_monitoring(self) -> None:
        """Start continuous health monitoring."""
        self.health_monitor.start_monitoring()

    def stop_health_monitoring(self) -> None:
        """Stop continuous health monitoring."""
        self.health_monitor.stop_monitoring()

    def get_deployment_health_status(
        self, deployment_id: str
    ) -> dict[str, Any]:
        """Get health status of a deployment.

        Args:
            deployment_id: Deployment identifier

        Returns:
            Health status information
        """
        return self.health_monitor.get_deployment_status(deployment_id)

    def get_all_health_statuses(self) -> dict[str, dict[str, Any]]:
        """Get health status of all monitored deployments.

        Returns:
            Dictionary mapping deployment IDs to their health status
        """
        return self.health_monitor.get_all_deployment_statuses()

    def deploy_with_strategy(
        self,
        config: DeploymentConfig,
        strategy: DeploymentStrategy,
        deployment_func: Callable[..., DeploymentResult],
        **kwargs,
    ) -> DeploymentResult:
        """Deploy using specified strategy with rollback capabilities.

        Args:
            config: Deployment configuration
            strategy: Deployment strategy to use
            deployment_func: Function to perform actual deployment
            **kwargs: Additional deployment parameters

        Returns:
            Deployment result with rollback information
        """
        deployment_id = f"{config.artifact_id}-{int(time.time())}"
        metadata = DeploymentMetadata(
            deployment_id=deployment_id,
            artifact_id=config.artifact_id,
            strategy=strategy,
            state=DeploymentState.PENDING,
            start_time=time.time(),
        )

        # Find previous deployment for rollback
        previous_deployment_id = self._find_current_deployment(
            config.target_environment
        )
        if previous_deployment_id:
            metadata.previous_deployment_id = previous_deployment_id

        self.logger.info(
            f"Starting {strategy.value} deployment: {deployment_id}"
        )

        try:
            # Execute strategy-specific deployment
            if strategy == DeploymentStrategy.BLUE_GREEN:
                result = self._blue_green_deploy(
                    config, deployment_func, metadata, **kwargs
                )
            elif strategy == DeploymentStrategy.CANARY:
                result = self._canary_deploy(
                    config, deployment_func, metadata, **kwargs
                )
            elif strategy == DeploymentStrategy.ROLLING:
                result = self._rolling_deploy(
                    config, deployment_func, metadata, **kwargs
                )
            else:  # RECREATE
                result = self._recreate_deploy(
                    config, deployment_func, metadata, **kwargs
                )

            metadata.state = DeploymentState.SUCCESS
            metadata.end_time = time.time()

            # Start performance monitoring
            if result.success and result.deployment_url:
                self._start_performance_monitoring(
                    deployment_id, result.deployment_url
                )

            # Start health monitoring if health check URL is available
            if result.success and result.health_check_url:
                self.add_deployment_monitoring(
                    deployment_id=deployment_id,
                    health_check_url=result.health_check_url,
                    process_name=config.artifact_id,
                    check_interval=30,
                )
                metadata.health_check_url = result.health_check_url

            # Send success alerts
            self._send_alerts("deployment_success", metadata, result=result)

            return result

        except Exception as e:
            self.logger.error(f"Deployment failed: {e}")
            metadata.state = DeploymentState.FAILED
            metadata.end_time = time.time()

            # Attempt automatic rollback
            rollback_result = self._attempt_rollback(metadata, e)
            if rollback_result:
                metadata.state = DeploymentState.ROLLED_BACK
                metadata.rollback_reason = str(e)

            # Send failure alerts
            self._send_alerts("deployment_failure", metadata, error=str(e))

            return DeploymentResult(
                success=False,
                deployment_id=deployment_id,
                artifact_id=config.artifact_id,
                target_environment=config.target_environment,
                error_message=str(e),
            )

    def _blue_green_deploy(
        self,
        config: DeploymentConfig,
        deployment_func: Callable[..., DeploymentResult],
        metadata: DeploymentMetadata,
        **kwargs,
    ) -> DeploymentResult:
        """Execute blue-green deployment strategy.

        Args:
            config: Deployment configuration
            deployment_func: Deployment function
            metadata: Deployment metadata
            **kwargs: Additional parameters

        Returns:
            Deployment result
        """
        self.logger.info("Executing blue-green deployment")

        # Find current active deployment (green)
        current_deployment = self._find_current_deployment(
            config.target_environment
        )
        if current_deployment:
            metadata.previous_deployment_id = current_deployment

        # Deploy to inactive environment (blue)
        metadata.state = DeploymentState.IN_PROGRESS
        blue_result = deployment_func(config, environment="blue", **kwargs)

        if not blue_result.success:
            raise RuntimeError(
                f"Blue deployment failed: {blue_result.error_message}"
            )

        # Health check blue deployment
        metadata.state = DeploymentState.HEALTH_CHECKING
        if blue_result.health_check_url:
            if not self.health_checker.wait_for_healthy(
                blue_result.health_check_url
            ):
                raise RuntimeError("Blue deployment health check failed")

        # Switch traffic to blue
        self._switch_traffic(config.target_environment, "blue")

        # Health check after traffic switch
        if blue_result.health_check_url:
            if not self.health_checker.wait_for_healthy(
                blue_result.health_check_url
            ):
                # Rollback to green if health check fails
                self._switch_traffic(config.target_environment, "green")
                raise RuntimeError("Health check failed after traffic switch")

        # Decommission old green deployment
        if current_deployment:
            self._decommission_deployment(current_deployment)

        return blue_result

    def _canary_deploy(
        self,
        config: DeploymentConfig,
        deployment_func: Callable[..., DeploymentResult],
        metadata: DeploymentMetadata,
        **kwargs,
    ) -> DeploymentResult:
        """Execute canary deployment strategy.

        Args:
            config: Deployment configuration
            deployment_func: Deployment function
            metadata: Deployment metadata
            **kwargs: Additional parameters

        Returns:
            Deployment result
        """
        self.logger.info("Executing canary deployment")

        # Deploy canary with small traffic percentage
        metadata.state = DeploymentState.IN_PROGRESS
        canary_result = deployment_func(
            config, environment="canary", traffic_percentage=10, **kwargs
        )

        if not canary_result.success:
            raise RuntimeError(
                f"Canary deployment failed: {canary_result.error_message}"
            )

        # Health check canary
        metadata.state = DeploymentState.HEALTH_CHECKING
        if canary_result.health_check_url:
            if not self.health_checker.wait_for_healthy(
                canary_result.health_check_url
            ):
                raise RuntimeError("Canary health check failed")

        # Monitor canary performance
        if not self._monitor_canary_performance(canary_result, **kwargs):
            raise RuntimeError("Canary performance monitoring failed")

        # Gradually increase traffic
        for percentage in [25, 50, 75, 100]:
            if canary_result.deployment_url:
                self._update_traffic_split(
                    canary_result.deployment_url, percentage
                )
            time.sleep(30)  # Wait between traffic increases

            if canary_result.health_check_url:
                if not self.health_checker.check_health(
                    canary_result.health_check_url
                ):
                    raise RuntimeError(
                        f"Health check failed at {percentage}% traffic"
                    )

        return canary_result

    def _rolling_deploy(
        self,
        config: DeploymentConfig,
        deployment_func: Callable[..., DeploymentResult],
        metadata: DeploymentMetadata,
        **kwargs,
    ) -> DeploymentResult:
        """Execute rolling deployment strategy.

        Args:
            config: Deployment configuration
            deployment_func: Deployment function
            metadata: Deployment metadata
            **kwargs: Additional parameters

        Returns:
            Deployment result
        """
        self.logger.info("Executing rolling deployment")

        # Get current deployment info
        current_replicas = self._get_current_replicas(
            config.target_environment
        )
        if current_replicas == 0:
            # No current deployment, do simple deploy
            metadata.state = DeploymentState.IN_PROGRESS
            return deployment_func(config, **kwargs)

        # Rolling update: gradually replace replicas
        metadata.state = DeploymentState.IN_PROGRESS
        for i in range(current_replicas):
            # Deploy one replica at a time
            replica_result = deployment_func(
                config,
                replica_index=i,
                total_replicas=current_replicas,
                **kwargs,
            )

            if not replica_result.success:
                raise RuntimeError(f"Replica {i} deployment failed")

            # Health check new replica
            if replica_result.health_check_url:
                if not self.health_checker.wait_for_healthy(
                    replica_result.health_check_url
                ):
                    raise RuntimeError(f"Replica {i} health check failed")

            # Remove old replica
            self._remove_old_replica(config.target_environment, i)

        return replica_result

    def _recreate_deploy(
        self,
        config: DeploymentConfig,
        deployment_func: Callable[..., DeploymentResult],
        metadata: DeploymentMetadata,
        **kwargs,
    ) -> DeploymentResult:
        """Execute recreate deployment strategy.

        Args:
            config: Deployment configuration
            deployment_func: Deployment function
            metadata: Deployment metadata
            **kwargs: Additional parameters

        Returns:
            Deployment result
        """
        self.logger.info("Executing recreate deployment")

        # Remove old deployment
        self._remove_current_deployment(config.target_environment)

        # Deploy new version
        metadata.state = DeploymentState.IN_PROGRESS
        result = deployment_func(config, **kwargs)

        if not result.success:
            raise RuntimeError(
                f"Recreate deployment failed: {result.error_message}"
            )

        # Health check new deployment
        metadata.state = DeploymentState.HEALTH_CHECKING
        if result.health_check_url:
            if not self.health_checker.wait_for_healthy(
                result.health_check_url
            ):
                raise RuntimeError("Recreate deployment health check failed")

        return result

    def _attempt_rollback(
        self, metadata: DeploymentMetadata, error: Exception
    ) -> bool:
        """Attempt automatic rollback on deployment failure.

        Args:
            metadata: Deployment metadata
            error: Error that caused failure

        Returns:
            True if rollback was successful
        """
        self.logger.info(
            f"Attempting rollback for deployment {metadata.deployment_id}"
        )

        if not metadata.previous_deployment_id:
            self.logger.warning("No previous deployment to rollback to")
            return False

        try:
            metadata.state = DeploymentState.ROLLING_BACK

            # Switch traffic back to previous deployment
            self._switch_traffic(metadata.target_environment, "previous")

            # Verify rollback health
            previous_metadata = self.deployment_history.get(
                metadata.previous_deployment_id
            )
            if previous_metadata and previous_metadata.health_check_url:
                if not self.health_checker.wait_for_healthy(
                    previous_metadata.health_check_url
                ):
                    self.logger.error("Rollback health check failed")
                    return False

            self.logger.info("Rollback completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            return False

    def manual_rollback(self, deployment_id: str) -> bool:
        """Manually rollback a deployment.

        Args:
            deployment_id: ID of deployment to rollback

        Returns:
            True if rollback was successful
        """
        metadata = self.deployment_history.get(deployment_id)
        if not metadata:
            self.logger.error(f"Deployment {deployment_id} not found")
            return False

        return self._attempt_rollback(metadata, Exception("Manual rollback"))

    def get_deployment_status(self, deployment_id: str) -> dict[str, Any]:
        """Get deployment status and metadata.

        Args:
            deployment_id: Deployment ID

        Returns:
            Deployment status information
        """
        metadata = self.deployment_history.get(deployment_id)
        if not metadata:
            return {"error": "Deployment not found"}

        return {
            "deployment_id": metadata.deployment_id,
            "artifact_id": metadata.artifact_id,
            "strategy": metadata.strategy.value,
            "state": metadata.state.value,
            "start_time": metadata.start_time,
            "end_time": metadata.end_time,
            "duration": (metadata.end_time or time.time())
            - metadata.start_time,
            "previous_deployment_id": metadata.previous_deployment_id,
            "rollback_reason": metadata.rollback_reason,
            "health_check_url": metadata.health_check_url,
            "metrics_url": metadata.metrics_url,
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
        for metadata in self.deployment_history.values():
            if artifact_id and metadata.artifact_id != artifact_id:
                continue
            history.append(self.get_deployment_status(metadata.deployment_id))

        return sorted(history, key=lambda x: x["start_time"], reverse=True)

    def get_performance_metrics(self, deployment_id: str) -> dict[str, Any]:
        """Get performance metrics for deployment.

        Args:
            deployment_id: Deployment ID

        Returns:
            Performance metrics dictionary
        """
        if deployment_id in self.performance_monitors:
            return self.performance_monitors[deployment_id].get_metrics()
        return {}

    def stop_performance_monitoring(self, deployment_id: str) -> None:
        """Stop performance monitoring for deployment.

        Args:
            deployment_id: Deployment ID to stop monitoring
        """
        if deployment_id in self.performance_monitors:
            self.performance_monitors[deployment_id].stop_monitoring()
            del self.performance_monitors[deployment_id]
            self.logger.info(
                f"Stopped performance monitoring for {deployment_id}"
            )

    # Helper methods for deployment strategies
    def _find_current_deployment(self, environment: str) -> str | None:
        """Find current active deployment for environment."""
        # Implementation would query deployment platform
        return None

    def _switch_traffic(self, environment: str, target: str) -> None:
        """Switch traffic to target deployment."""
        self.logger.info(f"Switching traffic to {target} in {environment}")

    def _decommission_deployment(self, deployment_id: str) -> None:
        """Decommission old deployment."""
        self.logger.info(f"Decommissioning deployment {deployment_id}")

    def _monitor_canary_performance(
        self, result: DeploymentResult, **kwargs
    ) -> bool:
        """Monitor canary deployment performance."""
        # Implementation would check metrics
        return True

    def _update_traffic_split(
        self, deployment_url: str, percentage: int
    ) -> None:
        """Update traffic split for canary deployment."""
        self.logger.info(f"Updating traffic split to {percentage}%")

    def _get_current_replicas(self, environment: str) -> int:
        """Get current number of replicas."""
        # Implementation would query deployment platform
        return 3

    def _remove_old_replica(self, environment: str, index: int) -> None:
        """Remove old replica during rolling deployment."""
        self.logger.info(f"Removing old replica {index}")

    def _remove_current_deployment(self, environment: str) -> None:
        """Remove current deployment for recreate strategy."""
        self.logger.info(f"Removing current deployment in {environment}")

    def _start_performance_monitoring(
        self, deployment_id: str, deployment_url: str
    ) -> None:
        """Start performance monitoring for deployment.

        Args:
            deployment_id: Deployment ID to monitor
            deployment_url: Deployment URL for monitoring
        """
        try:
            monitor = PerformanceMonitor(deployment_url)
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


class PerformanceMonitor:
    """Monitor deployment performance metrics."""

    def __init__(self, deployment_url: str) -> None:
        """Initialize performance monitor.

        Args:
            deployment_url: URL of deployment to monitor
        """
        self.deployment_url = deployment_url
        self.logger = logging.getLogger(__name__)
        self.metrics: dict[str, list[float]] = {
            "response_time": [],
            "throughput": [],
            "error_rate": [],
            "memory_usage": [],
            "cpu_usage": [],
        }
        self.monitoring = False
        self.monitor_thread = None

    def start_monitoring(self, deployment_id: str) -> None:
        """Start performance monitoring.

        Args:
            deployment_id: Deployment ID being monitored
        """
        self.monitoring = True
        self.logger.info(f"Started monitoring deployment {deployment_id}")

        import threading
        import time

        def monitor_loop():
            while self.monitoring:
                try:
                    self._collect_metrics()
                    time.sleep(30)  # Collect metrics every 30 seconds
                except Exception as e:
                    self.logger.warning(f"Error collecting metrics: {e}")
                    time.sleep(60)  # Wait longer on error

        self.monitor_thread = threading.Thread(
            target=monitor_loop, daemon=True
        )
        self.monitor_thread.start()

    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("Stopped performance monitoring")

    def _collect_metrics(self) -> None:
        """Collect performance metrics."""
        try:
            import time

            import requests

            # Measure response time
            start_time = time.time()
            requests.get(f"{self.deployment_url}/healthz", timeout=10)
            response_time = (time.time() - start_time) * 1000  # Convert to ms

            self.metrics["response_time"].append(response_time)

            # Simulate other metrics (in real implementation, these would come
            # from monitoring system)
            self.metrics["throughput"].append(100.0)  # requests/second
            self.metrics["error_rate"].append(0.01)  # 1% error rate
            self.metrics["memory_usage"].append(512.0)  # MB
            self.metrics["cpu_usage"].append(25.0)  # percentage

            # Keep only last 100 measurements
            for metric_name in self.metrics:
                if len(self.metrics[metric_name]) > 100:
                    self.metrics[metric_name] = self.metrics[metric_name][
                        -100:
                    ]

        except Exception as e:
            self.logger.warning(f"Failed to collect metrics: {e}")

    def get_metrics(self) -> dict[str, Any]:
        """Get current performance metrics.

        Returns:
            Dictionary with performance metrics
        """
        if not self.metrics["response_time"]:
            return {}

        return {
            "current": {
                "response_time_ms": self.metrics["response_time"][-1],
                "throughput_rps": self.metrics["throughput"][-1],
                "error_rate": self.metrics["error_rate"][-1],
                "memory_usage_mb": self.metrics["memory_usage"][-1],
                "cpu_usage_percent": self.metrics["cpu_usage"][-1],
            },
            "average": {
                "response_time_ms": sum(self.metrics["response_time"])
                / len(self.metrics["response_time"]),
                "throughput_rps": sum(self.metrics["throughput"])
                / len(self.metrics["throughput"]),
                "error_rate": sum(self.metrics["error_rate"])
                / len(self.metrics["error_rate"]),
                "memory_usage_mb": sum(self.metrics["memory_usage"])
                / len(self.metrics["memory_usage"]),
                "cpu_usage_percent": sum(self.metrics["cpu_usage"])
                / len(self.metrics["cpu_usage"]),
            },
            "trends": {
                "response_time_trend": self._calculate_trend(
                    self.metrics["response_time"]
                ),
                "throughput_trend": self._calculate_trend(
                    self.metrics["throughput"]
                ),
                "error_rate_trend": self._calculate_trend(
                    self.metrics["error_rate"]
                ),
            },
        }

    def _calculate_trend(self, values: list[float]) -> str:
        """Calculate trend direction.

        Args:
            values: List of metric values

        Returns:
            Trend direction: "improving", "degrading", or "stable"
        """
        if len(values) < 10:
            return "stable"

        recent_avg = sum(values[-10:]) / 10
        older_avg = (
            sum(values[-20:-10]) / 10 if len(values) >= 20 else values[0]
        )

        if recent_avg > older_avg * 1.1:
            return "degrading"
        elif recent_avg < older_avg * 0.9:
            return "improving"
        else:
            return "stable"


class AlertHandler:
    """Base class for alert handlers."""

    def send_alert(
        self, alert_type: str, metadata: DeploymentMetadata, **kwargs
    ) -> None:
        """Send alert.

        Args:
            alert_type: Type of alert
            metadata: Deployment metadata
            **kwargs: Additional alert data
        """
        raise NotImplementedError


class LoggingAlertHandler(AlertHandler):
    """Alert handler that logs alerts."""

    def __init__(self) -> None:
        """Initialize logging alert handler."""
        self.logger = logging.getLogger(__name__)

    def send_alert(
        self, alert_type: str, metadata: DeploymentMetadata, **kwargs
    ) -> None:
        """Send alert via logging.

        Args:
            alert_type: Type of alert
            metadata: Deployment metadata
            **kwargs: Additional alert data
        """
        if alert_type == "deployment_success":
            duration = (metadata.end_time or time.time()) - metadata.start_time
            self.logger.info(
                f"✅ Deployment {metadata.deployment_id} completed "
                f"successfully in {duration:.2f}s"
            )
        elif alert_type == "deployment_failure":
            error_msg = kwargs.get("error", "Unknown error")
            self.logger.error(
                f"❌ Deployment {metadata.deployment_id} failed: {error_msg}"
            )
        elif alert_type == "rollback_triggered":
            self.logger.warning(
                f"🔄 Rollback triggered for deployment "
                f"{metadata.deployment_id}: {metadata.rollback_reason}"
            )


class EmailAlertHandler(AlertHandler):
    """Alert handler that sends email alerts."""

    def __init__(
        self, smtp_server: str, smtp_port: int, username: str, password: str
    ) -> None:
        """Initialize email alert handler.

        Args:
            smtp_server: SMTP server address
            smtp_port: SMTP server port
            username: Email username
            password: Email password
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.logger = logging.getLogger(__name__)

    def send_alert(
        self, alert_type: str, metadata: DeploymentMetadata, **kwargs
    ) -> None:
        """Send alert via email.

        Args:
            alert_type: Type of alert
            metadata: Deployment metadata
            **kwargs: Additional alert data
        """
        try:
            import smtplib
            from email.mime.multipart import MIMEMultipart
            from email.mime.text import MIMEText

            # Create email message
            msg = MIMEMultipart()
            msg["From"] = self.username
            msg["To"] = "admin@crackseg.com"  # Configure recipient
            msg["Subject"] = (
                f"Deployment Alert: {alert_type.replace('_', ' ').title()}"
            )

            # Create email body
            duration = (metadata.end_time or time.time()) - metadata.start_time
            body = f"""
Deployment Alert: {alert_type.replace("_", " ").title()}

Deployment ID: {metadata.deployment_id}
Artifact ID: {metadata.artifact_id}
Strategy: {metadata.strategy.value}
Status: {metadata.state.value}
Duration: {duration:.2f}s

"""

            if alert_type == "deployment_failure":
                error_msg = kwargs.get("error", "Unknown error")
                body += f"Error: {error_msg}\n"
            elif alert_type == "rollback_triggered":
                body += f"Rollback Reason: {metadata.rollback_reason}\n"

            msg.attach(MIMEText(body, "plain"))

            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)
            server.quit()

            self.logger.info(f"Email alert sent for {alert_type}")

        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")


class SlackAlertHandler(AlertHandler):
    """Alert handler that sends Slack alerts."""

    def __init__(self, webhook_url: str) -> None:
        """Initialize Slack alert handler.

        Args:
            webhook_url: Slack webhook URL
        """
        self.webhook_url = webhook_url
        self.logger = logging.getLogger(__name__)

    def send_alert(
        self, alert_type: str, metadata: DeploymentMetadata, **kwargs
    ) -> None:
        """Send alert via Slack.

        Args:
            alert_type: Type of alert
            metadata: Deployment metadata
            **kwargs: Additional alert data
        """
        try:
            import json

            import requests

            # Create Slack message
            duration = (metadata.end_time or time.time()) - metadata.start_time
            if alert_type == "deployment_success":
                color = "good"
                emoji = "✅"
                title = "Deployment Successful"
            elif alert_type == "deployment_failure":
                color = "danger"
                emoji = "❌"
                title = "Deployment Failed"
            elif alert_type == "rollback_triggered":
                color = "warning"
                emoji = "🔄"
                title = "Rollback Triggered"
            else:
                color = "good"
                emoji = "ℹ️"
                title = "Deployment Alert"

            message = {
                "attachments": [
                    {
                        "color": color,
                        "title": f"{emoji} {title}",
                        "fields": [
                            {
                                "title": "Deployment ID",
                                "value": metadata.deployment_id,
                                "short": True,
                            },
                            {
                                "title": "Artifact ID",
                                "value": metadata.artifact_id,
                                "short": True,
                            },
                            {
                                "title": "Strategy",
                                "value": metadata.strategy.value,
                                "short": True,
                            },
                            {
                                "title": "Status",
                                "value": metadata.state.value,
                                "short": True,
                            },
                            {
                                "title": "Duration",
                                "value": f"{duration:.2f}s",
                                "short": True,
                            },
                        ],
                    }
                ]
            }

            if alert_type == "deployment_failure":
                error_msg = kwargs.get("error", "Unknown error")
                message["attachments"][0]["fields"].append(
                    {"title": "Error", "value": error_msg, "short": False}
                )

            # Send to Slack
            response = requests.post(
                self.webhook_url,
                data=json.dumps(message),
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                self.logger.info(f"Slack alert sent for {alert_type}")
            else:
                self.logger.warning(
                    f"Failed to send Slack alert: {response.status_code}"
                )

        except Exception as e:
            self.logger.error(f"Failed to send Slack alert: {e}")
