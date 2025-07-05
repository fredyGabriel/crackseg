"""Training state management for CrackSeg GUI.

This module provides training state management capabilities including
status tracking, metrics storage, and training lifecycle management.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class TrainingState:
    """Manage training state and metrics for the GUI."""

    def __init__(self) -> None:
        """Initialize training state."""
        self._is_running = False
        self._is_paused = False
        self._current_epoch = 0
        self._total_epochs = 0
        self._current_metrics: dict[str, Any] = {}
        self._training_history: list[dict[str, Any]] = []

    @property
    def is_running(self) -> bool:
        """Check if training is currently running."""
        return self._is_running

    @property
    def is_paused(self) -> bool:
        """Check if training is currently paused."""
        return self._is_paused

    @property
    def current_epoch(self) -> int:
        """Get current training epoch."""
        return self._current_epoch

    @property
    def total_epochs(self) -> int:
        """Get total number of epochs."""
        return self._total_epochs

    @property
    def current_metrics(self) -> dict[str, Any]:
        """Get current training metrics."""
        return self._current_metrics.copy()

    def start_training(self, total_epochs: int = 100) -> None:
        """Start training process.

        Args:
            total_epochs: Total number of epochs to train
        """
        self._is_running = True
        self._is_paused = False
        self._current_epoch = 0
        self._total_epochs = total_epochs
        self._current_metrics = {}
        logger.info(f"Training started for {total_epochs} epochs")

    def pause_training(self) -> None:
        """Pause training process."""
        if self._is_running:
            self._is_paused = True
            logger.info("Training paused")

    def resume_training(self) -> None:
        """Resume paused training."""
        if self._is_paused:
            self._is_paused = False
            logger.info("Training resumed")

    def stop_training(self) -> None:
        """Stop training process."""
        self._is_running = False
        self._is_paused = False
        logger.info("Training stopped")

    def update_metrics(self, metrics: dict[str, Any]) -> None:
        """Update current training metrics.

        Args:
            metrics: Dictionary of metric name to value
        """
        # Calculate deltas for display
        for key, value in metrics.items():
            if key in self._current_metrics:
                delta_key = f"{key}_delta"
                delta = value - self._current_metrics[key]
                metrics[delta_key] = delta

        self._current_metrics.update(metrics)

    def advance_epoch(self) -> None:
        """Advance to next epoch."""
        if self._is_running:
            self._current_epoch += 1

            # Save current metrics to history
            epoch_data = {
                "epoch": self._current_epoch,
                "metrics": self._current_metrics.copy(),
            }
            self._training_history.append(epoch_data)

    def get_training_history(self) -> list[dict[str, Any]]:
        """Get complete training history."""
        return self._training_history.copy()

    def reset_state(self) -> None:
        """Reset training state to initial values."""
        self._is_running = False
        self._is_paused = False
        self._current_epoch = 0
        self._total_epochs = 0
        self._current_metrics = {}
        self._training_history = []
        logger.info("Training state reset")
