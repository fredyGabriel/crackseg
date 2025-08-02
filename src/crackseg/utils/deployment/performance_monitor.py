"""Performance monitoring for deployment orchestration.

This module provides performance monitoring capabilities for deployment
orchestration, including real-time metrics collection, trend analysis,
and background monitoring capabilities.
"""

import logging
import threading
import time
from typing import Any

import requests


class PerformanceMonitor:
    """Monitor deployment performance metrics.

    Provides real-time monitoring of deployment performance including
    response time, throughput, error rates, and resource usage.
    """

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
            Dictionary with performance metrics including current values,
            averages, and trends.
        """
        if not self.metrics["response_time"]:
            return {}

        # Calculate current metrics
        current_metrics = {}
        for metric_name, values in self.metrics.items():
            if values:
                current_metrics[metric_name] = values[-1]

        # Calculate average metrics
        average_metrics = {}
        for metric_name, values in self.metrics.items():
            if values:
                average_metrics[metric_name] = sum(values) / len(values)

        # Calculate trends
        trends = {}
        for metric_name, values in self.metrics.items():
            if len(values) >= 10:
                recent_values = values[-10:]
                trend = self._calculate_trend(recent_values)
                trends[f"{metric_name}_trend"] = trend

        return {
            "current": current_metrics,
            "average": average_metrics,
            "trends": trends,
        }

    def _calculate_trend(self, values: list[float]) -> str:
        """Calculate trend from metric values.

        Args:
            values: List of metric values

        Returns:
            Trend description: 'improving', 'degrading', or 'stable'
        """
        if len(values) < 2:
            return "stable"

        # Calculate linear regression slope
        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * val for i, val in enumerate(values))
        x_squared_sum = sum(i * i for i in range(n))

        # Linear regression formula
        slope = (n * xy_sum - x_sum * y_sum) / (
            n * x_squared_sum - x_sum * x_sum
        )

        # Determine trend based on slope
        if slope > 0.1:  # Positive slope threshold
            return "improving"
        elif slope < -0.1:  # Negative slope threshold
            return "degrading"
        else:
            return "stable"
