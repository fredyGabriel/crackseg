"""Progress tracking utilities for UI responsive operations.

This module provides progress tracking capabilities including progress updates,
callback mechanisms, and progress reporting for long-running background tasks.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class ProgressUpdate:
    """Progress update information for UI display.

    Provides comprehensive progress information that can be used
    to update UI components with task execution status.

    Attributes:
        current: Current progress value
        total: Total progress value (for percentage calculation)
        message: Human-readable progress message
        stage: Current stage name
        timestamp: When this update was created

    Example:
        >>> update = ProgressUpdate(50, 100, "Processing data", "DataLoad")
        >>> print(f"Progress: {update.percentage:.1f}%")
        Progress: 50.0%
    """

    current: float
    total: float
    message: str
    stage: str = "Processing"
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        """Set timestamp if not provided."""
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    @property
    def percentage(self) -> float:
        """Calculate progress percentage.

        Returns:
            Progress as percentage (0.0 to 100.0)
        """
        if self.total <= 0:
            return 0.0
        return min(100.0, (self.current / self.total) * 100.0)

    @property
    def is_complete(self) -> bool:
        """Check if progress indicates completion.

        Returns:
            True if current progress equals or exceeds total
        """
        return self.current >= self.total

    @property
    def elapsed_time(self) -> float:
        """Calculate elapsed time since update creation.

        Returns:
            Elapsed time in seconds
        """
        return time.time() - self.timestamp

    def to_dict(self) -> dict[str, Any]:
        """Convert progress update to dictionary format.

        Returns:
            Dictionary representation of progress update
        """
        return {
            "current": self.current,
            "total": self.total,
            "percentage": self.percentage,
            "message": self.message,
            "stage": self.stage,
            "timestamp": self.timestamp,
            "is_complete": self.is_complete,
        }


# Type alias for progress callback functions
type ProgressCallback = Callable[[ProgressUpdate], None]


class ProgressTracker:
    """Utility class for tracking and reporting progress updates.

    Provides a convenient interface for managing progress updates
    and notifying registered callbacks during task execution.

    Example:
        >>> tracker = ProgressTracker(total=100)
        >>> tracker.add_callback(
        ...     lambda update: print(f"{update.percentage:.1f}%")
        ... )
        >>> tracker.update(25, "Quarter complete")
        25.0%
    """

    def __init__(
        self, total: float = 100.0, stage: str = "Processing"
    ) -> None:
        """Initialize progress tracker.

        Args:
            total: Total progress value
            stage: Initial stage name
        """
        self._total = total
        self._current = 0.0
        self._stage = stage
        self._callbacks: list[ProgressCallback] = []

    def add_callback(self, callback: ProgressCallback) -> None:
        """Add a progress callback.

        Args:
            callback: Function to call on progress updates
        """
        self._callbacks.append(callback)

    def remove_callback(self, callback: ProgressCallback) -> None:
        """Remove a progress callback.

        Args:
            callback: Function to remove from callbacks
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def update(
        self, current: float, message: str, stage: str | None = None
    ) -> None:
        """Update progress and notify callbacks.

        Args:
            current: Current progress value
            message: Progress message
            stage: Stage name (uses current stage if None)
        """
        self._current = current
        if stage is not None:
            self._stage = stage

        update = ProgressUpdate(
            current=self._current,
            total=self._total,
            message=message,
            stage=self._stage,
        )

        for callback in self._callbacks:
            try:
                callback(update)
            except Exception:
                # Continue with other callbacks even if one fails
                pass

    def increment(self, amount: float = 1.0, message: str = "") -> None:
        """Increment progress by specified amount.

        Args:
            amount: Amount to increment
            message: Progress message
        """
        self.update(self._current + amount, message)

    def complete(self, message: str = "Completed") -> None:
        """Mark progress as complete.

        Args:
            message: Completion message
        """
        self.update(self._total, message)

    @property
    def current_progress(self) -> ProgressUpdate:
        """Get current progress as ProgressUpdate.

        Returns:
            Current progress update
        """
        return ProgressUpdate(
            current=self._current,
            total=self._total,
            message="Current progress",
            stage=self._stage,
        )
