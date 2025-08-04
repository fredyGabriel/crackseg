"""Metrics collection for monitoring system."""

import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import MetricsConfig, MonitoringResult, ResourceMetrics


class MetricsCollector:
    """Collect and aggregate monitoring metrics."""

    def __init__(self, config: "MetricsConfig") -> None:
        """Initialize metrics collector.

        Args:
            config: Metrics configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.metrics_history: dict[str, list[Any]] = {}

    def collect_metrics(
        self,
        health_result: "MonitoringResult",
        resource_metrics: "ResourceMetrics",
        performance_metrics: dict[str, Any],
    ) -> "MonitoringResult":
        """Collect and aggregate all metrics.

        Args:
            health_result: Health check result
            resource_metrics: System resource metrics
            performance_metrics: Performance metrics

        Returns:
            Aggregated monitoring result
        """
        timestamp = time.time()

        # Aggregate all metrics
        aggregated_metrics = {
            "health": {
                "status": health_result.health_status,
                "response_time_ms": health_result.metrics.get(
                    "response_time_ms", 0.0
                ),
            },
            "resources": {
                "cpu_usage_percent": resource_metrics.cpu_usage_percent,
                "memory_usage_mb": resource_metrics.memory_usage_mb,
                "disk_usage_percent": resource_metrics.disk_usage_percent,
                "network_io_mbps": resource_metrics.network_io_mbps,
            },
            "performance": performance_metrics,
        }

        # Store in history
        self._store_metrics(timestamp, aggregated_metrics)

        # Check for alerts
        alerts = self._check_alerts(aggregated_metrics)

        return MonitoringResult(
            success=health_result.success,
            timestamp=timestamp,
            metrics=aggregated_metrics,
            health_status=health_result.health_status,
            alerts=alerts,
            error_message=health_result.error_message,
        )

    def _store_metrics(
        self, timestamp: float, metrics: dict[str, Any]
    ) -> None:
        """Store metrics in history.

        Args:
            timestamp: Metrics timestamp
            metrics: Metrics data
        """
        # Store timestamp
        if "timestamps" not in self.metrics_history:
            self.metrics_history["timestamps"] = []
        self.metrics_history["timestamps"].append(timestamp)

        # Store individual metrics
        for category, category_metrics in metrics.items():
            for metric_name, value in category_metrics.items():
                full_name = f"{category}.{metric_name}"
                if full_name not in self.metrics_history:
                    self.metrics_history[full_name] = []
                self.metrics_history[full_name].append(value)

        # Clean up old data
        self._cleanup_old_metrics()

    def _cleanup_old_metrics(self) -> None:
        """Remove old metrics data."""
        current_time = time.time()
        cutoff_time = current_time - self.config.retention_period

        if "timestamps" in self.metrics_history:
            # Find indices to keep
            keep_indices = [
                i
                for i, ts in enumerate(self.metrics_history["timestamps"])
                if ts >= cutoff_time
            ]

            # Keep only recent data
            for metric_name in self.metrics_history:
                if len(self.metrics_history[metric_name]) > len(keep_indices):
                    self.metrics_history[metric_name] = [
                        self.metrics_history[metric_name][i]
                        for i in keep_indices
                    ]

    def _check_alerts(self, metrics: dict[str, Any]) -> list[str]:
        """Check for alert conditions.

        Args:
            metrics: Current metrics

        Returns:
            List of alert messages
        """
        alerts = []

        # Health alerts
        if "health" in metrics:
            health_metrics = metrics["health"]
            if health_metrics["status"] != "healthy":
                alerts.append(
                    f"Health check failed: {health_metrics['status']}"
                )

            if health_metrics["response_time_ms"] > 1000:
                alerts.append(
                    f"High response time: {health_metrics['response_time_ms']:.1f}ms"
                )

        # Resource alerts
        if "resources" in metrics:
            resource_metrics = metrics["resources"]
            if resource_metrics["cpu_usage_percent"] > 80:
                alerts.append(
                    f"High CPU usage: {resource_metrics['cpu_usage_percent']:.1f}%"
                )

            if resource_metrics["memory_usage_mb"] > 2048:
                alerts.append(
                    f"High memory usage: {resource_metrics['memory_usage_mb']:.1f}MB"
                )

            if resource_metrics["disk_usage_percent"] > 90:
                alerts.append(
                    f"High disk usage: {resource_metrics['disk_usage_percent']:.1f}%"
                )

        # Performance alerts
        if "performance" in metrics:
            perf_metrics = metrics["performance"]
            for metric_name, metric_data in perf_metrics.items():
                if isinstance(metric_data, dict) and "current" in metric_data:
                    current_value = metric_data["current"]
                    if metric_name == "error_rate" and current_value > 0.05:
                        alerts.append(f"High error rate: {current_value:.2%}")
                    elif metric_name == "throughput" and current_value < 10:
                        alerts.append(
                            f"Low throughput: {current_value:.1f} RPS"
                        )

        return alerts

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get summary of collected metrics.

        Returns:
            Metrics summary
        """
        if (
            not self.metrics_history
            or "timestamps" not in self.metrics_history
        ):
            return {}

        summary = {
            "total_data_points": len(self.metrics_history["timestamps"]),
            "time_range": {
                "start": min(self.metrics_history["timestamps"]),
                "end": max(self.metrics_history["timestamps"]),
            },
            "metrics": {},
        }

        # Calculate statistics for each metric
        for metric_name, values in self.metrics_history.items():
            if metric_name != "timestamps" and values:
                summary["metrics"][metric_name] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "average": sum(values) / len(values),
                    "latest": values[-1] if values else None,
                }

        return summary

    def export_metrics(self, format: str = "json") -> str:
        """Export metrics data.

        Args:
            format: Export format ("json", "csv")

        Returns:
            Exported metrics data
        """
        if format == "json":
            import json

            return json.dumps(self.metrics_history, indent=2)
        elif format == "csv":
            import csv
            import io

            output = io.StringIO()
            writer = csv.writer(output)

            # Write header
            if self.metrics_history:
                headers = ["timestamp"] + list(self.metrics_history.keys())
                writer.writerow(headers)

                # Write data
                timestamps = self.metrics_history.get("timestamps", [])
                for i, timestamp in enumerate(timestamps):
                    row = [timestamp]
                    for metric_name in self.metrics_history:
                        if metric_name != "timestamps":
                            values = self.metrics_history[metric_name]
                            row.append(values[i] if i < len(values) else "")
                    writer.writerow(row)

            return output.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format}")
