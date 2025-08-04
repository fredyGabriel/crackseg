"""Simplified resource alerting system with threshold-based monitoring.

This module provides streamlined alerting capabilities that integrate with
the ResourceMonitor and ThresholdChecker for resource monitoring.
"""

import logging

from ..resources.config import ThresholdConfig
from ..resources.monitor import ResourceMonitor
from ..resources.snapshot import ResourceDict
from .checker import ThresholdChecker
from .types import Alert, AlertCallback, AlertType

logger = logging.getLogger(__name__)


class AlertingSystem:
    """Simplified alerting system for resource monitoring.

    Integrates ResourceMonitor with ThresholdChecker to provide
    comprehensive alerting for resource exhaustion and threshold violations.

    Features:
    - Real-time threshold monitoring via ThresholdChecker
    - Alert callback system
    - Alert history and resolution tracking
    - Integration with existing monitoring framework
    - RTX 3070 Ti specific optimizations

    Example:
        >>> monitor = ResourceMonitor()
        >>> alerting = AlertingSystem(monitor)
        >>> alerting.add_callback(lambda alert: print(alert.message))
        >>> alerting.start_monitoring()
    """

    def __init__(
        self,
        resource_monitor: ResourceMonitor,
        threshold_config: ThresholdConfig | None = None,
    ) -> None:
        """Initialize the alerting system.

        Args:
            resource_monitor: ResourceMonitor instance to monitor
            threshold_config: Threshold configuration for alerting
        """
        self.resource_monitor = resource_monitor
        self.threshold_config = threshold_config or ThresholdConfig()

        # Initialize threshold checker
        self.threshold_checker = ThresholdChecker(self.threshold_config)

        # Alert management
        self._callbacks: list[AlertCallback] = []
        self._active_alerts: dict[str, Alert] = {}
        self._alert_history: list[Alert] = []
        self._max_history = 1000

        # Monitoring state
        self._monitoring_active = False

    def add_callback(self, callback: AlertCallback) -> None:
        """Add callback to be executed when alerts are triggered."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: AlertCallback) -> None:
        """Remove callback from execution list."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def start_monitoring(self) -> None:
        """Start real-time alerting by adding callback to resource monitor."""
        if self._monitoring_active:
            logger.warning("Alerting system already active")
            return

        # Add our callback to the resource monitor
        self.resource_monitor.add_callback(self._check_thresholds)
        self._monitoring_active = True

        # Establish baseline for leak detection
        try:
            current_snapshot = self.resource_monitor.get_current_snapshot()
            self.threshold_checker.establish_baseline(
                current_snapshot.to_dict()
            )
        except Exception as e:
            logger.warning(f"Could not establish baseline: {e}")

        logger.info("Alerting system started")

    def stop_monitoring(self) -> None:
        """Stop real-time alerting."""
        if not self._monitoring_active:
            return

        self.resource_monitor.remove_callback(self._check_thresholds)
        self._monitoring_active = False
        logger.info("Alerting system stopped")

    def get_active_alerts(self) -> list[Alert]:
        """Get list of currently active (unresolved) alerts."""
        return [
            alert
            for alert in self._active_alerts.values()
            if not alert.resolved
        ]

    def get_alert_history(self, count: int | None = None) -> list[Alert]:
        """Get historical alerts.

        Args:
            count: Number of recent alerts to return (None for all)

        Returns:
            List of alerts ordered by timestamp (most recent first)
        """
        history = sorted(
            self._alert_history, key=lambda a: a.timestamp, reverse=True
        )
        return history[:count] if count else history

    def clear_alert_history(self) -> None:
        """Clear alert history."""
        self._alert_history.clear()

    def resolve_alert(
        self, alert_type: AlertType, resource_name: str = ""
    ) -> bool:
        """Manually resolve an active alert.

        Args:
            alert_type: Type of alert to resolve
            resource_name: Specific resource name (optional)

        Returns:
            True if alert was resolved, False if not found
        """
        alert_key = f"{alert_type.value}_{resource_name}"
        if alert_key in self._active_alerts:
            self._active_alerts[alert_key].resolve()
            logger.info(f"Manually resolved alert: {alert_key}")
            return True
        return False

    def _check_thresholds(self, resource_dict: ResourceDict) -> None:
        """Check thresholds and trigger alerts."""
        try:
            # Get alerts from threshold checker
            new_alerts = self.threshold_checker.check_all_thresholds(
                resource_dict
            )

            # Process each alert
            for alert in new_alerts:
                self._process_alert(alert)

            # Auto-resolve alerts that are no longer active
            self._auto_resolve_alerts(resource_dict)

        except Exception as e:
            logger.error(f"Error in threshold checking: {e}")

    def _process_alert(self, alert: Alert) -> None:
        """Process a new alert."""
        alert_key = f"{alert.alert_type.value}_{alert.resource_name}"

        # Don't trigger duplicate alerts
        if (
            alert_key in self._active_alerts
            and not self._active_alerts[alert_key].resolved
        ):
            return

        # Store alert
        self._active_alerts[alert_key] = alert
        self._alert_history.append(alert)

        # Limit history size
        if len(self._alert_history) > self._max_history:
            self._alert_history.pop(0)

        # Execute callbacks
        for callback in self._callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

        # Log alert
        level = (
            logging.CRITICAL
            if alert.severity.value == "critical"
            else logging.WARNING
        )
        logger.log(
            level, f"ALERT [{alert.severity.value.upper()}]: {alert.message}"
        )

    def _auto_resolve_alerts(self, resource_dict: ResourceDict) -> None:
        """Auto-resolve alerts that are no longer triggered."""
        # This is a simplified version - in practice you'd check if conditions
        # that triggered the alert are still present
        for alert_key, alert in self._active_alerts.items():
            if not alert.resolved:
                # Simple logic: resolve if value is now below threshold
                current_value = self._get_current_value_for_alert(
                    alert, resource_dict
                )
                if (
                    current_value is not None
                    and current_value < alert.threshold
                ):
                    alert.resolve()
                    logger.info(f"Auto-resolved alert: {alert_key}")

    def _get_current_value_for_alert(
        self, alert: Alert, resource_dict: ResourceDict
    ) -> float | None:
        """Get current value for alert comparison."""
        # Map alert types to resource dict keys
        value_mapping = {
            "cpu_usage": "cpu_percent",
            "memory_usage": "memory_percent",
            "gpu_memory_usage": "gpu_memory_used_mb",
            "gpu_utilization": "gpu_utilization_percent",
            "gpu_temperature": "gpu_temperature_celsius",
            "process_count": "process_count",
            "file_handles": "file_handles",
            "thread_count": "thread_count",
            "temp_files_accumulation": "temp_files_size_mb",
            "network_connections": "network_connections",
        }

        key = value_mapping.get(alert.alert_type.value)
        if key:
            return resource_dict.get(key)
        return None
