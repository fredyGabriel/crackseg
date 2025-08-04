"""
Core components for the callback-based monitoring system.

This module defines the base classes and handlers for creating and managing
a pipeline of monitoring callbacks that can be attached to a training or
evaluation loop.
"""

import time
from typing import Any

from ..manager import MonitoringManager


class BaseCallback:
    """
    Abstract base class for all monitoring callbacks.

    Callbacks can be used to inject custom logic at different stages of
    a training or evaluation process (e.g., collecting metrics, logging,
    early stopping).
    """

    def __init__(self) -> None:
        self.metrics_manager: MonitoringManager | None = None

    def set_manager(self, manager: MonitoringManager) -> None:
        """Sets the MonitoringManager for the callback to use."""
        self.metrics_manager = manager

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        """Called at the beginning of training."""
        pass

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        """Called at the end of training."""
        pass

    def on_epoch_begin(
        self, epoch: int, logs: dict[str, Any] | None = None
    ) -> None:
        """Called at the beginning of an epoch."""
        pass

    def on_epoch_end(
        self, epoch: int, logs: dict[str, Any] | None = None
    ) -> None:
        """Called at the end of an epoch."""
        pass

    def on_batch_begin(
        self, batch: int, logs: dict[str, Any] | None = None
    ) -> None:
        """Called at the beginning of a training batch."""
        pass

    def on_batch_end(
        self, batch: int, logs: dict[str, Any] | None = None
    ) -> None:
        """Called at the end of a training batch."""
        pass


class CallbackHandler:
    """
    Manages a list of callbacks and triggers their corresponding methods.
    """

    def __init__(
        self, callbacks: list[BaseCallback], manager: MonitoringManager
    ):
        """
        Initializes the CallbackHandler.

        Args:
            callbacks: A list of callback instances.
            manager: The MonitoringManager to be used by all callbacks.
        """
        self.callbacks = callbacks
        self.metrics_manager = manager
        for cb in self.callbacks:
            cb.set_manager(self.metrics_manager)

    def __getattr__(self, name: str) -> Any:
        """
        Dynamically creates a method that calls the same method on all
        callbacks.
        """

        def handler(*args: Any, **kwargs: Any) -> None:
            for callback in self.callbacks:
                try:
                    getattr(callback, name)(*args, **kwargs)
                except Exception as e:
                    # Log error but don't stop execution
                    print(
                        f"Warning: Callback {callback.__class__.__name__} "
                        f"failed in {name}: {e}"
                    )

        return handler


class TimerCallback(BaseCallback):
    """A simple callback to measure and log timing metrics."""

    def __init__(self) -> None:
        super().__init__()
        self._start_time: float = 0
        self._epoch_start_time: float = 0
        self._batch_start_time: float = 0

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        """Records the start time of the training."""
        self._start_time = time.time()

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        """Calculates and logs the total training duration."""
        if self.metrics_manager:
            duration = time.time() - self._start_time
            self.metrics_manager.log({"train_duration_sec": duration})

    def on_epoch_begin(
        self, epoch: int, logs: dict[str, Any] | None = None
    ) -> None:
        """Records the start time of an epoch."""
        self._epoch_start_time = time.time()
        if self.metrics_manager:
            self.metrics_manager.current_epoch = epoch

    def on_epoch_end(
        self, epoch: int, logs: dict[str, Any] | None = None
    ) -> None:
        """Calculates and logs the epoch duration."""
        if self.metrics_manager:
            duration = time.time() - self._epoch_start_time
            self.metrics_manager.log({"epoch_duration_sec": duration})

    def on_batch_begin(
        self, batch: int, logs: dict[str, Any] | None = None
    ) -> None:
        """Records the start time of a batch."""
        self._batch_start_time = time.time()
        if self.metrics_manager:
            self.metrics_manager.current_step += 1

    def on_batch_end(
        self, batch: int, logs: dict[str, Any] | None = None
    ) -> None:
        """Calculates and logs the batch duration."""
        if self.metrics_manager:
            duration = time.time() - self._batch_start_time
            self.metrics_manager.log({"batch_duration_ms": duration * 1000})
