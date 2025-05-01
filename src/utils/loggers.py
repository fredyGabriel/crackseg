"""Logging utilities for training and evaluation."""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any

# Conditional import for TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    _TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    _TENSORBOARD_AVAILABLE = False


def get_logger(name: str, level: str = 'INFO') -> logging.Logger:
    """Return a configured logger with the given name and level.

    This function creates and configures a standard Python logger for general
    application logging (status messages, errors, etc.).

    Args:
        name: The name for the logger, typically the module name.
        level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level.upper())
    return logger


class BaseLogger(ABC):
    """Abstract base class for metrics and training loggers."""

    @abstractmethod
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value."""
        pass

    @abstractmethod
    def log_config(self, config: Dict[str, Any]) -> None:
        """Log the experiment configuration."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the logger and release resources."""
        pass


class TensorBoardLogger(BaseLogger):
    """Logger for TensorBoard."""

    def __init__(self, log_dir: str):
        """Initialize TensorBoard SummaryWriter.

        Args:
            log_dir: Directory to save TensorBoard logs.
        """
        if not _TENSORBOARD_AVAILABLE:
            raise ImportError(
                "TensorBoard not found. Please install it with "
                "'pip install tensorboard' or add it to your environment."
            )
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logs will be saved to: {log_dir}")

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value to TensorBoard."""
        self.writer.add_scalar(tag, value, step)

    def log_config(self, config: Dict[str, Any]) -> None:
        """Log hyperparameters/config to TensorBoard.

        Note: TensorBoard expects Dict[str, Union[str, bool, int, float]].
        Complex objects or nested dicts might not render well.
        We can log the config as text or use add_hparams.
        For simplicity, logging as text for now.
        """
        # Option 1: Log as text (simple)
        # config_str = str(config) # Or format nicely
        # self.writer.add_text("config", config_str)

        # Option 2: Use add_hparams (more structured but restrictive)
        # Needs flattening and filtering of the config dict.
        # Placeholder - implementation depends on config structure.
        # hparams = self._flatten_config(config)
        # self.writer.add_hparams(hparam_dict=hparams, metric_dict={})
        pass  # Config logging needs more refinement based on usage

    def close(self) -> None:
        """Close the SummaryWriter."""
        self.writer.close()

    # Helper for add_hparams (if needed later)
    # def _flatten_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
    #     # Implement logic to flatten and filter config for add_hparams
    #     pass


class NoOpLogger(BaseLogger):
    """A logger that performs no operations."""

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        pass

    def log_config(self, config: Dict[str, Any]) -> None:
        pass

    def close(self) -> None:
        pass


def log_metrics_dict(
    logger: BaseLogger,
    metrics: Dict[str, float],
    step: int,
    prefix: str = ''
) -> None:
    """Logs a dictionary of scalar metrics using the provided logger.

    Args:
        logger: An instance of BaseLogger (e.g., TensorBoardLogger).
        metrics: Dictionary where keys are metric names (str)
                 and values are scalar numbers (int or float).
        step: The logging step (e.g., epoch number or global step).
        prefix: Optional prefix to add to metric tags (e.g., 'train/').
    """
    if not logger:
        return  # Do nothing if logger is None or NoOpLogger

    for name, value in metrics.items():
        # Only log numeric types (int, float), explicitly excluding bool
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            tag = f"{prefix}{name}" if prefix else name
            try:
                # Ensure value is float for logger
                logger.log_scalar(tag=tag, value=float(value), step=step)
            except Exception as e:
                # Log a warning if a specific metric fails to log
                standard_logger = get_logger(__name__)
                standard_logger.warning(
                    f"Failed to log metric '{tag}' at step {step}: {e}"
                )
        # else: # Optionally log a warning for non-numeric types
        #     standard_logger = get_logger(__name__)
        #     standard_logger.debug(f"Skipping non-numeric metric '{name}'...")


# --- Factory Function --- (To be added later or handled by Hydra directly)
# def create_logger(cfg: DictConfig) -> BaseLogger:
#     logger_type = cfg.logging.get("type", None)
#     log_dir = cfg.logging.get("log_dir", "outputs/tensorboard")

#     if logger_type == "tensorboard":
#         return TensorBoardLogger(log_dir=log_dir)
#     elif logger_type is None or logger_type == "none":
#         print("Logging disabled.")
#         return NoOpLogger()
#     else:
#         raise ValueError(f"Unsupported logger type: {logger_type}")
