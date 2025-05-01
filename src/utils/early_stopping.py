"""Implements the EarlyStopping callback logic."""

import numpy as np
from typing import Optional

from src.utils.loggers import get_logger

# Logger for this module
logger = get_logger(__name__)


class EarlyStopping:
    """Monitors a metric and stops training when it stops improving."""

    def __init__(
        self,
        monitor_metric: str = "loss",
        patience: int = 10,
        min_delta: float = 0,
        mode: str = "min",
        verbose: bool = True
    ):
        """Initializes the EarlyStopping callback.

        Args:
            monitor_metric: Name of the metric to monitor (from val results).
            patience: Number of epochs with no improvement after which
                      training will be stopped.
            min_delta: Minimum change in the monitored quantity to qualify as
                       an improvement.
            mode: One of {"min", "max"}. In "min" mode, training stops when the
                  quantity monitored has stopped decreasing; in "max" mode
                  it stops when the quantity monitored has stopped increasing.
            verbose: If True, prints a message for each improvement or stop.
        """
        if mode not in ["min", "max"]:
            raise ValueError(f"EarlyStopping mode '{mode}' is unknown, choose \
from [\"min\", \"max\"].")

        self.monitor_metric = monitor_metric
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score = np.inf if self.mode == "min" else -np.inf
        self.early_stop = False

        # Adjust delta based on mode
        self.min_delta *= 1 if self.mode == "min" else -1

        log_msg = (
            f"EarlyStopping initialized: monitor='{self.monitor_metric}', "
            f"patience={self.patience}, mode='{self.mode}', "
            f"min_delta={abs(self.min_delta)}"
        )
        logger.info(log_msg)

    def step(self, current_metric_value: Optional[float]) -> bool:
        """Checks if training should stop based on the current metric value.

        Args:
            current_metric_value: The latest value of the monitored metric.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        if current_metric_value is None:
            logger.warning(
                f"Early stopping condition metric '{self.monitor_metric}' "
                f"was not found. Skipping check."
            )
            return False

        score = current_metric_value

        # Check for improvement
        improvement = False
        if self.mode == "min":
            if score < self.best_score - self.min_delta:
                improvement = True
        else:  # mode == "max"
            if score + self.min_delta > self.best_score:
                improvement = True

        if improvement:
            if self.verbose:
                logger.info(
                    f"EarlyStopping metric '{self.monitor_metric}' improved "
                    f"from {self.best_score:.4f} to {score:.4f}."
                )
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"EarlyStopping counter: {self.counter} out of "
                    f"{self.patience} for metric '{self.monitor_metric}'"
                )
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info(
                        f"Early stopping triggered after {self.patience} "
                        f"epochs with no improvement."
                    )

        return self.early_stop

    def should_stop(self) -> bool:
        """Returns the current early stopping status."""
        return self.early_stop
