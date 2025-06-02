import logging
import os
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig

from src.utils.core.exceptions import ConfigError
from src.utils.experiment.manager import ExperimentManager

log = logging.getLogger(__name__)


def initialize_experiment(
    cfg: DictConfig, base_dir: str | None = None
) -> tuple[str, Any]:
    """Initialize experiment directory and logging.

    Creates experiment directory structure using ExperimentManager
    and initializes the experiment logger.

    Args:
        cfg: The configuration object
        base_dir: Base directory for experiments (if None, uses 'outputs')

    Returns:
        Tuple containing:
        - experiment_dir: Path to the experiment directory
        - logger: Initialized logger instance

    Raises:
        ConfigError: If there are issues with the configuration
    """
    try:
        # Lazy import to avoid circular dependency
        from src.utils.logging.experiment import ExperimentLogger

        # Use 'outputs' as base_dir if none provided, ensuring all experiments
        # are in a consistent location
        if base_dir is None:
            # Get project root directory
            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            base_dir = os.path.join(project_root, "outputs")
            log.info(f"Using standard outputs directory as base: {base_dir}")

        # Make sure the base_dir exists
        os.makedirs(base_dir, exist_ok=True)

        # Get experiment name from config, or use a default
        experiment_name: str = "default"
        if hasattr(cfg, "experiment") and hasattr(cfg.experiment, "name"):
            experiment_name = cast(str, cfg.experiment.name)

        # Leer constantes de timestamp desde la configuración
        ts_cfg = cast(dict[str, Any], cfg.get("timestamp_parsing", {}))
        min_parts: int = cast(int, ts_cfg.get("min_parts", 2))
        date_len: int = cast(int, ts_cfg.get("date_len", 8))
        time_len: int = cast(int, ts_cfg.get("time_len", 6))

        # Extract timestamp from current directory if possible
        timestamp: str | None = None
        current_dir = Path(os.getcwd())
        if current_dir.name.startswith("20") and "-" in current_dir.name:
            # Try to extract timestamp (YYYYMMDD-HHMMSS) from directory name
            parts = current_dir.name.split("-")
            if (
                len(parts) >= min_parts
                and len(parts[0]) == date_len
                and len(parts[1]) == time_len
            ):
                timestamp = f"{parts[0]}-{parts[1]}"
                log.info(
                    f"Extracted timestamp from directory name: {timestamp}"
                )

        # Create experiment manager
        manager = ExperimentManager(
            base_dir=base_dir,
            experiment_name=experiment_name,
            config=cfg,
            create_dirs=True,
            timestamp=timestamp,  # Pass the extracted timestamp if available
        )

        # Get experiment directory
        experiment_dir = str(manager.experiment_dir)
        log.info(f"Experiment directory: {experiment_dir}")

        # Initialize experiment logger using the manager
        logger = ExperimentLogger(
            log_dir=base_dir,
            experiment_name=experiment_name,
            config=cfg,
            log_level=cast(str, cfg.get("log_level", "INFO")),
            log_to_file=cast(bool, cfg.get("log_to_file", True)),
        )

        return experiment_dir, logger

    except Exception as e:
        raise ConfigError(f"Failed to initialize experiment: {str(e)}") from e
