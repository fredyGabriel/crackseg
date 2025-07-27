"""Early stopping implementation for training."""

from collections.abc import Callable

import numpy as np

from crackseg.utils.logging import get_logger

logger = get_logger(__name__)


class EarlyStopping:
    """Early stopping to prevent overfitting.

    Stops training when a monitored metric has not improved for a given
    number of epochs.
    """

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        mode: str = "min",
        verbose: bool = True,
    ):
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: One of {'min', 'max'}. In 'min' mode, stops when the
                quantity stops decreasing; in 'max' mode when it stops
                increasing.
            verbose: If True, prints message when early stopping triggers
        """
        self.patience: int = patience
        self.min_delta: float = abs(min_delta)
        self.mode: str = mode
        self.verbose: bool = verbose
        self.counter: int = 0
        self.best_value: float | None = None
        self.early_stop: bool = False

        if mode not in ["min", "max"]:
            raise ValueError(f"mode '{mode}' is unknown")

        self.monitor_op: Callable[[float, float], bool] = (
            np.less if mode == "min" else np.greater
        )
        # Adjust delta based on mode for comparison
        self.delta_val: float = (
            -self.min_delta if mode == "min" else self.min_delta
        )
        logger.info(
            f"Initialized early stopping (mode={mode}, patience={patience})"
        )

    def should_stop(self, current_value: float | None) -> bool:
        """Check if training should stop (alias for __call__).

        This method provides trainer compatibility.

        Args:
            current_value: Current value of the monitored metric.

        Returns:
            True if training should stop, False otherwise.
        """
        return self.__call__(current_value)

    def __call__(self, current_value: float | None) -> bool:
        """Check if training should stop.

        Args:
            current_value: Current value of the monitored metric. If None,
                           it's ignored, and the counter doesn't increase.

        Returns:
            True if training should stop, False otherwise
        """
        # Handle None: Don't update counter or stop if metric is missing
        if current_value is None:
            logger.debug("Early stopping received None metric, ignoring step.")
            return False

        # First valid value initializes best_value
        if self.best_value is None:
            self.best_value = current_value
            logger.debug(
                f"Early stopping initialized best_value= \
{self.best_value:.4f}"
            )
            return False

        # Check for improvement including delta
        # Mode min: stop if current_value >= best_value - min_delta
        # Mode max: stop if current_value <= best_value + min_delta
        # Using monitor_op: improve if op(current_value, best_value + delta_val
        # )
        if self.monitor_op(current_value, self.best_value + self.delta_val):
            # Improvement detected
            improve_msg = (
                f"Improvement detected: {current_value:.4f} vs best "
                f"{self.best_value:.4f} (delta={self.delta_val:.4f})"
            )
            logger.debug(improve_msg)
            self.best_value = current_value
            self.counter = 0
        else:
            # No improvement
            self.counter += 1
            if self.verbose:
                counter_msg = (
                    f"Early stopping counter: {self.counter}/{self.patience}"
                )
                logger.info(counter_msg)
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info("Early stopping triggered")
                return True
        return False

    def reset(self) -> None:
        """Reset the early stopping state."""
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        logger.debug("Early stopping state reset")
