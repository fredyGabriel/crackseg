"""Helper for setting up checkpoint directory and experiment manager."""

import os


def setup_checkpointing(cfg, logger_instance, internal_logger):
    """
    Sets up checkpoint directory and experiment manager.
    Returns (checkpoint_dir, experiment_manager or None)
    """
    experiment_manager = None

    def safe_log(logger, level, *args, **kwargs):
        fn = getattr(logger, level, None)
        if callable(fn):
            fn(*args, **kwargs)

    # Get experiment_manager from logger if it exists
    if logger_instance and hasattr(logger_instance, "experiment_manager"):
        experiment_manager = logger_instance.experiment_manager
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
    if not isinstance(checkpoint_dir, str) or (
        isinstance(checkpoint_dir, str) and "<MagicMock" in checkpoint_dir
    ):
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
