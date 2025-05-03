import os
import logging
from typing import Tuple, Optional
from pathlib import Path

from omegaconf import DictConfig

from src.utils import ExperimentLogger, ConfigError
from src.utils.experiment_manager import ExperimentManager

log = logging.getLogger(__name__)


def initialize_experiment(
    cfg: DictConfig,
    base_dir: Optional[str] = None
) -> Tuple[str, ExperimentLogger]:
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
        # Use 'outputs' as base_dir if none provided, ensuring all experiments
        # are in a consistent location
        if base_dir is None:
            # Get project root directory
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))
            base_dir = os.path.join(project_root, "outputs")
            log.info(f"Using standard outputs directory as base: {base_dir}")

        # Make sure the base_dir exists
        os.makedirs(base_dir, exist_ok=True)

        # Get experiment name from config, or use a default
        experiment_name = "default"
        if hasattr(cfg, "experiment") and hasattr(cfg.experiment, "name"):
            experiment_name = cfg.experiment.name

        # Extract timestamp from current directory if possible
        timestamp = None
        current_dir = Path(os.getcwd())
        if current_dir.name.startswith("20") and "-" in current_dir.name:
            # Try to extract timestamp (YYYYMMDD-HHMMSS) from directory name
            parts = current_dir.name.split("-")
            if len(parts) >= 2 and len(parts[0]) == 8 and len(parts[1]) == 6:
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
            timestamp=timestamp  # Pass the extracted timestamp if available
        )

        # Get experiment directory
        experiment_dir = str(manager.experiment_dir)
        log.info(f"Experiment directory: {experiment_dir}")

        # Initialize experiment logger using the manager
        logger = ExperimentLogger(
            log_dir=base_dir,
            experiment_name=experiment_name,
            config=cfg,
            log_level=cfg.get("log_level", "INFO"),
            log_to_file=cfg.get("log_to_file", True)
        )

        return experiment_dir, logger

    except Exception as e:
        raise ConfigError(
            f"Failed to initialize experiment: {str(e)}"
        ) from e
