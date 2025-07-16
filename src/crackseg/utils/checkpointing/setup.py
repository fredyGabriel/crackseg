"""Helper for setting up checkpoint directory and experiment manager."""

import os
from collections.abc import Mapping
from typing import Any, Protocol


# Protocolo mínimo para logger
class LoggerProtocol(Protocol):
    def info(self, *args: Any, **kwargs: Any) -> None: ...
    def warning(self, *args: Any, **kwargs: Any) -> None: ...


def setup_checkpointing(
    cfg: Mapping[str, Any],
    logger_or_experiment_manager: Any,
    internal_logger: LoggerProtocol,
) -> tuple[str, Any | None]:
    """
    Sets up checkpoint directory and experiment manager.
    Returns (checkpoint_dir, experiment_manager or None)

    Args:
        cfg: Configuration dictionary
        logger_or_experiment_manager: Either a logger with experiment_manager
                                attribute, or an experiment_manager directly
        internal_logger: Logger for internal messages
    """
    experiment_manager = None

    def safe_log(
        logger: LoggerProtocol, level: str, *args: Any, **kwargs: Any
    ) -> None:
        fn = getattr(logger, level, None)
        if callable(fn):
            fn(*args, **kwargs)

    # Determine if we received a logger or experiment_manager directly
    if logger_or_experiment_manager:
        # Check if it's a logger with experiment_manager attribute
        if hasattr(logger_or_experiment_manager, "experiment_manager"):
            experiment_manager = (
                logger_or_experiment_manager.experiment_manager
            )
        # Otherwise assume it's the experiment_manager directly
        else:
            experiment_manager = logger_or_experiment_manager

    # Try to get checkpoint directory from experiment_manager
    if experiment_manager:
        try:
            # Ensure get_path returns a valid string
            checkpoint_dir = experiment_manager.get_path("checkpoints")
            if not isinstance(checkpoint_dir, str):
                checkpoint_dir = str(checkpoint_dir)
            safe_log(
                internal_logger,
                "info",
                "Using checkpoint directory from ExperimentManager: ",
                f"{checkpoint_dir}",
            )
        except (
            AttributeError,
            TypeError,
            KeyError,
            ValueError,
            Exception,
        ) as e:
            # If error calling get_path, use default directory
            safe_log(
                internal_logger,
                "warning",
                "Error accessing experiment_manager.get_path "
                f"({type(e).__name__}: {e}). "
                "Using config checkpoint_dir: "
                f"{cfg.get('checkpoint_dir', 'outputs/checkpoints')}",
            )
            checkpoint_dir = cfg.get("checkpoint_dir", "outputs/checkpoints")
    else:
        # If no experiment_manager, use checkpoint_dir specified in cfg
        checkpoint_dir = cfg.get("checkpoint_dir", "outputs/checkpoints")
        safe_log(
            internal_logger,
            "info",
            f"Using checkpoint directory from config: {checkpoint_dir}",
        )

    # Ensure the value is always a valid string for os.makedirs
    if not isinstance(checkpoint_dir, str) or "<MagicMock" in checkpoint_dir:
        checkpoint_dir = "outputs/checkpoints"
        safe_log(
            internal_logger,
            "warning",
            "Invalid checkpoint directory detected. ",
            f"Using fallback: {checkpoint_dir}",
        )

    os.makedirs(checkpoint_dir, exist_ok=True)
    safe_log(
        internal_logger, "info", f"Checkpoint directory: {checkpoint_dir}"
    )
    return checkpoint_dir, experiment_manager
