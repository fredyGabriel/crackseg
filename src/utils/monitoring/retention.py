"""
Data retention policies for the monitoring framework.

This module provides configurable data retention policies to manage
memory usage and storage of monitoring metrics over time.
"""

import json
import time
from pathlib import Path
from typing import Protocol

from .manager import HistoryDict, MetricValue


class RetentionPolicy(Protocol):
    """Protocol for implementing data retention policies."""

    def should_retain(self, timestamp: float, current_time: float) -> bool:
        """
        Determine if data should be retained based on timestamp.

        Args:
            timestamp: The timestamp of the data point
            current_time: The current timestamp

        Returns:
            True if data should be retained, False otherwise
        """
        ...

    def get_description(self) -> str:
        """Get a human-readable description of the policy."""
        ...


class TimeBasedRetentionPolicy:
    """Retention policy based on time duration."""

    def __init__(self, max_age_seconds: float) -> None:
        """
        Initialize time-based retention policy.

        Args:
            max_age_seconds: Maximum age of data in seconds before removal
        """
        self.max_age_seconds = max_age_seconds

    def should_retain(self, timestamp: float, current_time: float) -> bool:
        """Check if data should be retained based on age."""
        age = current_time - timestamp
        return age <= self.max_age_seconds

    def get_description(self) -> str:
        """Get policy description."""
        return f"TimeBasedRetentionPolicy(max_age={self.max_age_seconds}s)"


class CountBasedRetentionPolicy:
    """Retention policy based on maximum number of data points."""

    def __init__(self, max_count: int) -> None:
        """
        Initialize count-based retention policy.

        Args:
            max_count: Maximum number of data points to retain
        """
        self.max_count = max_count

    def should_retain(self, timestamp: float, current_time: float) -> bool:
        """Always retain - count-based filtering done elsewhere."""
        return True

    def get_description(self) -> str:
        """Get policy description."""
        return f"CountBasedRetentionPolicy(max_count={self.max_count})"


class CompositeRetentionPolicy:
    """Composite retention policy combining multiple policies."""

    def __init__(self, policies: list[RetentionPolicy]) -> None:
        """
        Initialize composite retention policy.

        Args:
            policies: List of retention policies to combine
        """
        self.policies = policies

    def should_retain(self, timestamp: float, current_time: float) -> bool:
        """Data is retained if ALL policies agree to retain it."""
        return all(
            policy.should_retain(timestamp, current_time)
            for policy in self.policies
        )

    def get_description(self) -> str:
        """Get policy description."""
        descriptions = [policy.get_description() for policy in self.policies]
        return f"CompositeRetentionPolicy({', '.join(descriptions)})"


class RetentionManager:
    """
    Manages data retention for monitoring metrics.

    Provides functionality to apply retention policies, persist data,
    and clean up old metrics based on configurable policies.
    """

    def __init__(
        self,
        retention_policy: RetentionPolicy | None = None,
        storage_path: Path | None = None,
        auto_cleanup_interval: float = 300.0,  # 5 minutes
    ) -> None:
        """
        Initialize retention manager.

        Args:
            retention_policy: Policy for data retention
            storage_path: Path for persistent storage
            auto_cleanup_interval: Interval for automatic cleanup in seconds
        """
        self.retention_policy = retention_policy or TimeBasedRetentionPolicy(
            3600.0
        )  # 1 hour default
        self.storage_path = storage_path
        self.auto_cleanup_interval = auto_cleanup_interval
        self._last_cleanup_time = time.time()

    def apply_retention_policy(self, history: HistoryDict) -> HistoryDict:
        """
        Apply retention policy to history data.

        Args:
            history: Current metrics history

        Returns:
            Filtered history with retention policy applied
        """
        if not history:
            return history

        current_time = time.time()
        from collections import defaultdict

        filtered_history: HistoryDict = defaultdict(list)

        # Group metrics by base name (without _values, _steps, _timestamps
        # suffix)
        metric_groups = self._group_metrics(history)

        for base_name, group in metric_groups.items():
            if f"{base_name}_timestamps" not in group:
                # If no timestamps, keep all data
                filtered_history.update(group)
                continue

            timestamps = group[f"{base_name}_timestamps"]

            # Apply retention policy
            if isinstance(self.retention_policy, CountBasedRetentionPolicy):
                # Handle count-based policy
                max_count = self.retention_policy.max_count
                if len(timestamps) > max_count:
                    # Keep only the most recent entries
                    keep_indices = list(
                        range(len(timestamps) - max_count, len(timestamps))
                    )
                else:
                    keep_indices = list(range(len(timestamps)))
            else:
                # Handle time-based or composite policies
                keep_indices = [
                    i
                    for i, ts in enumerate(timestamps)
                    if self.retention_policy.should_retain(ts, current_time)
                ]

            # Filter all related arrays
            for key, values in group.items():
                if keep_indices:
                    filtered_history[key] = [values[i] for i in keep_indices]
                else:
                    filtered_history[key] = []

        return filtered_history

    def _group_metrics(
        self, history: HistoryDict
    ) -> dict[str, dict[str, list[MetricValue]]]:
        """Group metrics by base name."""
        groups: dict[str, dict[str, list[MetricValue]]] = {}

        for key, values in history.items():
            # Extract base name (remove _values, _steps, _timestamps suffix)
            if key.endswith(("_values", "_steps", "_timestamps")):
                base_name = key.rsplit("_", 1)[0]
            else:
                base_name = key

            if base_name not in groups:
                groups[base_name] = {}
            groups[base_name][key] = values

        return groups

    def should_cleanup(self) -> bool:
        """Check if automatic cleanup should be performed."""
        current_time = time.time()
        return (
            current_time - self._last_cleanup_time
        ) >= self.auto_cleanup_interval

    def mark_cleanup_done(self) -> None:
        """Mark that cleanup has been performed."""
        self._last_cleanup_time = time.time()

    def save_to_disk(self, history: HistoryDict) -> bool:
        """
        Save metrics history to disk.

        Args:
            history: Metrics history to save

        Returns:
            True if save was successful, False otherwise
        """
        if not self.storage_path:
            return False

        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert to JSON-serializable format
            serializable_history = {
                key: list(values) for key, values in history.items()
            }

            with open(self.storage_path, "w") as f:
                json.dump(serializable_history, f, indent=2)

            return True
        except Exception:
            return False

    def load_from_disk(self) -> HistoryDict:
        """
        Load metrics history from disk.

        Returns:
            Loaded metrics history or empty dict if loading fails
        """
        from collections import defaultdict

        if not self.storage_path or not self.storage_path.exists():
            return defaultdict(list)

        try:
            with open(self.storage_path) as f:
                data = json.load(f)

            # Convert back to defaultdict-like structure
            history: HistoryDict = defaultdict(list)
            history.update(data)
            return history
        except Exception:
            return defaultdict(list)

    def get_storage_size(self) -> int:
        """
        Get the size of stored data in bytes.

        Returns:
            Size in bytes, or 0 if no storage or error
        """
        if not self.storage_path or not self.storage_path.exists():
            return 0

        try:
            return self.storage_path.stat().st_size
        except Exception:
            return 0

    def clear_storage(self) -> bool:
        """
        Clear persistent storage.

        Returns:
            True if successful, False otherwise
        """
        if not self.storage_path or not self.storage_path.exists():
            return True

        try:
            self.storage_path.unlink()
            return True
        except Exception:
            return False
