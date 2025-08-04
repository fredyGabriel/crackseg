"""Performance monitoring for deployment system."""

import logging
import threading
import time
from typing import TYPE_CHECKING, Any

import requests

if TYPE_CHECKING:
    from .config import MetricsConfig


class PerformanceMonitor:
    """Monitor deployment performance metrics."""

    def __init__(self, deployment_url: str, config: "MetricsConfig") -> None:
        """Initialize performance monitor.

        Args:
            deployment_url: URL of deployment to monitor
            config: Metrics configuration
        """
        self.deployment_url = deployment_url
        self.config = config
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

        def monitor_loop():
            while self.monitoring:
                try:
                    self._collect_metrics()
                    time.sleep(self.config.collection_interval)
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

            # Keep only last N measurements
            for metric_name in self.metrics:
                if (
                    len(self.metrics[metric_name])
                    > self.config.max_data_points
                ):
                    self.metrics[metric_name] = self.metrics[metric_name][
                        -self.config.max_data_points :
                    ]

        except Exception as e:
            self.logger.warning(f"Failed to collect metrics: {e}")

    def get_metrics(self) -> dict[str, Any]:
        """Get current performance metrics.

        Returns:
            Dictionary of current metrics
        """
        current_metrics = {}
        for metric_name, values in self.metrics.items():
            if values:
                current_metrics[metric_name] = {
                    "current": values[-1],
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "trend": self._calculate_trend(values),
                }
            else:
                current_metrics[metric_name] = {
                    "current": 0.0,
                    "average": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "trend": "stable",
                }

        return current_metrics

    def _calculate_trend(self, values: list[float]) -> str:
        """Calculate trend of metric values.

        Args:
            values: List of metric values

        Returns:
            Trend: "increasing", "decreasing", or "stable"
        """
        if len(values) < 3:
            return "stable"

        # Calculate simple linear trend
        recent_values = values[-3:]
        if recent_values[0] < recent_values[1] < recent_values[2]:
            return "increasing"
        elif recent_values[0] > recent_values[1] > recent_values[2]:
            return "decreasing"
        else:
            return "stable"

    def get_alert_conditions(self) -> list[str]:
        """Get current alert conditions.

        Returns:
            List of alert conditions
        """
        alerts = []
        metrics = self.get_metrics()

        # Check response time
        if "response_time" in metrics:
            current_rt = metrics["response_time"]["current"]
            if current_rt > 1000:  # 1 second threshold
                alerts.append(f"High response time: {current_rt:.1f}ms")

        # Check error rate
        if "error_rate" in metrics:
            current_error = metrics["error_rate"]["current"]
            if current_error > 0.05:  # 5% threshold
                alerts.append(f"High error rate: {current_error:.2%}")

        # Check memory usage
        if "memory_usage" in metrics:
            current_memory = metrics["memory_usage"]["current"]
            if current_memory > 2048:  # 2GB threshold
                alerts.append(f"High memory usage: {current_memory:.1f}MB")

        return alerts
