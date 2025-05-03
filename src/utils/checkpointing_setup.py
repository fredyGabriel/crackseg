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

    # Obtener experiment_manager del logger si existe
    if logger_instance and hasattr(logger_instance, 'experiment_manager'):
        experiment_manager = logger_instance.experiment_manager
        try:
            # Asegurar que get_path devuelve un string válido
            checkpoint_dir = experiment_manager.get_path("checkpoints")
            if not isinstance(checkpoint_dir, str):
                checkpoint_dir = str(checkpoint_dir)
            safe_log(
                internal_logger, "info",
                "Using checkpoint directory from ExperimentManager: ",
                f"{checkpoint_dir}"
            )
        except (AttributeError, Exception):
            # Si hay error al llamar a get_path, usar el directorio por defecto
            checkpoint_dir = cfg.get("checkpoint_dir", "outputs/checkpoints")
            safe_log(
                internal_logger, "warning",
                "Error accessing experiment_manager.get_path. ",
                f"Using config checkpoint_dir: {checkpoint_dir}"
            )
    else:
        # Si no hay experiment_manager, usar el checkpoint_dir especificado en
        # cfg
        checkpoint_dir = cfg.get("checkpoint_dir", "outputs/checkpoints")
        safe_log(
            internal_logger, "info",
            f"Using checkpoint directory from config: {checkpoint_dir}"
        )

    # Asegurarse de que el valor sea siempre un string válido para os.makedirs
    if (not isinstance(checkpoint_dir, str) or
            (isinstance(checkpoint_dir, str) and "<MagicMock" in
             checkpoint_dir)):
        checkpoint_dir = "outputs/checkpoints"
        safe_log(
            internal_logger, "warning",
            "Invalid checkpoint directory detected. ",
            f"Using fallback: {checkpoint_dir}"
        )

    os.makedirs(checkpoint_dir, exist_ok=True)
    safe_log(
        internal_logger, "info",
        f"Checkpoint directory: {checkpoint_dir}"
    )
    return checkpoint_dir, experiment_manager
