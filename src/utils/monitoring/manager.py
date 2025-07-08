"""Monitoring Manager for collecting, storing, and reporting metrics."""

import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .retention import RetentionManager

MetricValue = int | float
MetricDict = dict[str, MetricValue]
HistoryDict = defaultdict[str, list[MetricValue]]


class MonitoringManager:
    """
    A central manager to handle metrics collection from various callbacks.

    This class provides a simple interface for callbacks to log metrics
    and stores them in a structured way, grouped by context (e.g., 'train',
    'validation') and execution step.
    """

    def __init__(
        self,
        retention_manager: Optional["RetentionManager"] = None,
        enable_persistence: bool = False,
        storage_path: Path | None = None,
    ) -> None:
        """
        Initializes the MonitoringManager.

        Args:
            retention_manager: Manager for data retention policies
            enable_persistence: Whether to enable persistent storage
            storage_path: Path for persistent storage
        """
        self._history: HistoryDict = defaultdict(list)
        self.current_step: int = 0
        self.current_epoch: int = 0
        self.context: str = "train"

        # Retention and persistence
        self.retention_manager = retention_manager
        self.enable_persistence = enable_persistence
        self.storage_path = storage_path

        # Load existing data if persistence is enabled
        if self.enable_persistence and self.retention_manager:
            self._load_persisted_data()

    def log(self, metrics: MetricDict, step: int | None = None) -> None:
        """
        Logs a dictionary of metrics at a specific step.

        Args:
            metrics: A dictionary of metric names to values.
            step: The step number for the log. If None, uses the internal
                step counter.
        """
        timestamp = time.time()
        log_step = step if step is not None else self.current_step

        for key, value in metrics.items():
            full_key = f"{self.context}/{key}"
            self._history[f"{full_key}_values"].append(value)
            self._history[f"{full_key}_steps"].append(log_step)
            self._history[f"{full_key}_timestamps"].append(timestamp)

    def set_context(self, context: str) -> None:
        """
        Sets the current logging context (e.g., 'train', 'val', 'test').

        Args:
            context: The name of the context.
        """
        self.context = context

    def get_history(self) -> dict[str, list[MetricValue]]:
        """
        Returns the complete history of all collected metrics.

        Returns:
            A dictionary containing lists of values, steps, and timestamps
            for each metric.
        """
        return dict(self._history)

    def get_last_metric(self, metric_name: str) -> MetricValue | None:
        """
        Retrieves the last logged value for a specific metric.

        Args:
            metric_name: The full name of the metric (e.g., 'train/loss').

        Returns:
            The last value of the metric, or None if not found.
        """
        key = f"{metric_name}_values"
        if key in self._history and self._history[key]:
            return self._history[key][-1]
        return None

    def reset(self) -> None:
        """Resets the metrics history."""
        self._history.clear()
        self.current_step = 0
        self.current_epoch = 0
        self.context = "train"

    def apply_retention_policy(self) -> None:
        """Apply retention policy to clean up old data."""
        if self.retention_manager:
            # Apply retention policy
            self._history = self.retention_manager.apply_retention_policy(
                self._history
            )

            # Save to disk if persistence is enabled
            if self.enable_persistence:
                self.retention_manager.save_to_disk(self._history)

            # Mark cleanup as done
            self.retention_manager.mark_cleanup_done()

    def should_cleanup(self) -> bool:
        """Check if automatic cleanup should be performed."""
        return (
            self.retention_manager is not None
            and self.retention_manager.should_cleanup()
        )

    def _load_persisted_data(self) -> None:
        """Load persisted data from disk."""
        if self.retention_manager:
            persisted_data = self.retention_manager.load_from_disk()
            if persisted_data:
                self._history.update(persisted_data)

    def save_to_disk(self) -> bool:
        """
        Save current metrics to disk.

        Returns:
            True if save was successful, False otherwise
        """
        if self.retention_manager and self.enable_persistence:
            return self.retention_manager.save_to_disk(self._history)
        return False

    def get_storage_info(self) -> dict[str, Any]:
        """
        Get information about data storage.

        Returns:
            Dictionary with storage information
        """
        info = {
            "persistence_enabled": self.enable_persistence,
            "storage_path": (
                str(self.storage_path) if self.storage_path else None
            ),
            "retention_policy": None,
            "storage_size_bytes": 0,
            "total_metrics": len(self._history),
        }

        if self.retention_manager:
            info["retention_policy"] = (
                self.retention_manager.retention_policy.get_description()
            )
            info["storage_size_bytes"] = (
                self.retention_manager.get_storage_size()
            )

        return info
