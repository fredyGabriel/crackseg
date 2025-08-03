"""Progress tracking for TensorBoard component startup."""

import time
from typing import Any


class ProgressTracker:
    """Tracks startup progress for TensorBoard component."""

    def __init__(self) -> None:
        """Initialize progress tracker."""
        self._start_time: float | None = None
        self._progress: float = 0.0
        self._completed: bool = False

    def start(self) -> None:
        """Start progress tracking."""
        self._start_time = time.time()
        self._progress = 0.1
        self._completed = False

    def update_progress(self, progress: float) -> None:
        """Update progress value.

        Args:
            progress: Progress value between 0.0 and 1.0.
        """
        self._progress = max(0.0, min(1.0, progress))

    def complete(self) -> None:
        """Mark progress as completed."""
        self._progress = 1.0
        self._completed = True

    def get_progress(self) -> float:
        """Get current progress value."""
        return self._progress

    def get_elapsed_time(self) -> float:
        """Get elapsed time since start."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    def is_completed(self) -> bool:
        """Check if progress is completed."""
        return self._completed

    def reset(self) -> None:
        """Reset progress tracker."""
        self._start_time = None
        self._progress = 0.0
        self._completed = False

    def to_dict(self) -> dict[str, Any]:
        """Export progress state as dictionary."""
        return {
            "start_time": self._start_time,
            "progress": self._progress,
            "completed": self._completed,
            "elapsed_time": self.get_elapsed_time(),
        }
