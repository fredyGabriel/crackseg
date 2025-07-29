"""
Experiment lifecycle management for ExperimentTracker.

This module provides lifecycle management methods for the ExperimentTracker
component.
"""

from collections.abc import Callable
from datetime import datetime

from crackseg.utils.experiment.metadata import ExperimentMetadata
from crackseg.utils.logging.base import get_logger

logger = get_logger(__name__)


class ExperimentLifecycleManager:
    """Manages experiment lifecycle operations."""

    def __init__(
        self, metadata: ExperimentMetadata, auto_save: bool = True
    ) -> None:
        """
        Initialize the lifecycle manager.

        Args:
            metadata: Experiment metadata to manage
            auto_save: Whether to automatically save metadata changes
        """
        self.metadata = metadata
        self.auto_save = auto_save

    def start_experiment(
        self, save_callback: Callable[[], None] | None = None
    ) -> None:
        """Mark experiment as started."""
        self.metadata.status = "running"
        self.metadata.started_at = datetime.now().isoformat()
        if self.auto_save and save_callback:
            save_callback()
        logger.info(f"Experiment started: {self.metadata.experiment_id}")

    def complete_experiment(
        self, save_callback: Callable[[], None] | None = None
    ) -> None:
        """Mark experiment as completed."""
        self.metadata.status = "completed"
        self.metadata.completed_at = datetime.now().isoformat()
        if self.auto_save and save_callback:
            save_callback()
        logger.info(f"Experiment completed: {self.metadata.experiment_id}")

    def fail_experiment(
        self,
        error_message: str = "",
        save_callback: Callable[[], None] | None = None,
    ) -> None:
        """Mark experiment as failed."""
        self.metadata.status = "failed"
        self.metadata.completed_at = datetime.now().isoformat()
        if error_message:
            self.metadata.description += f"\nError: {error_message}"
        if self.auto_save and save_callback:
            save_callback()
        logger.error(f"Experiment failed: {self.metadata.experiment_id}")

    def abort_experiment(
        self, save_callback: Callable[[], None] | None = None
    ) -> None:
        """Mark experiment as aborted."""
        self.metadata.status = "aborted"
        self.metadata.completed_at = datetime.now().isoformat()
        if self.auto_save and save_callback:
            save_callback()
        logger.warning(f"Experiment aborted: {self.metadata.experiment_id}")

    def update_training_progress(
        self,
        current_epoch: int,
        total_epochs: int,
        best_metrics: dict[str, float] | None = None,
        training_time_seconds: float | None = None,
        save_callback: Callable[[], None] | None = None,
    ) -> None:
        """
        Update training progress metadata.

        Args:
            current_epoch: Current training epoch
            total_epochs: Total number of epochs
            best_metrics: Best metrics achieved so far
            training_time_seconds: Total training time in seconds
            save_callback: Callback to save metadata
        """
        self.metadata.current_epoch = current_epoch
        self.metadata.total_epochs = total_epochs

        if best_metrics is not None:
            self.metadata.best_metrics.update(best_metrics)

        if training_time_seconds is not None:
            self.metadata.training_time_seconds = training_time_seconds

        if self.auto_save and save_callback:
            save_callback()
