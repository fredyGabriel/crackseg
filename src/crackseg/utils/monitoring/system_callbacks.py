"""Callbacks for monitoring system-level metrics like CPU and RAM."""

from typing import Any

import psutil

from .callbacks import BaseCallback
from .exceptions import MetricCollectionError


class SystemStatsCallback(BaseCallback):
    """
    Callback to collect and log system-level statistics.

    This callback uses `psutil` to monitor CPU utilization and memory usage.
    To minimize overhead, it's recommended to collect these metrics at less
    frequent events, such as `on_epoch_end`.
    """

    def on_epoch_end(
        self, epoch: int, logs: dict[str, Any] | None = None
    ) -> None:
        """Collects and logs CPU and memory stats at the end of an epoch."""
        if not self.metrics_manager:
            return

        try:
            # CPU Metrics
            cpu_percent = psutil.cpu_percent(interval=None)

            # Memory Metrics (Virtual Memory)
            mem_info = psutil.virtual_memory()
            mem_percent = mem_info.percent
            mem_used_gb = mem_info.used / (1024**3)

            metrics_to_log = {
                "cpu_util_percent": cpu_percent,
                "ram_util_percent": mem_percent,
                "ram_used_gb": mem_used_gb,
            }

            self.metrics_manager.log(metrics_to_log)

        except Exception as e:
            raise MetricCollectionError(
                f"Failed to collect system stats: {e}"
            ) from e
