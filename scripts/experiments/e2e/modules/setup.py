"""Setup module for end-to-end pipeline testing imports."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import project modules
try:
    from crackseg.data.factory import create_dataloaders_from_config
    from crackseg.utils import (
        load_checkpoint,
        save_checkpoint,
        set_random_seeds,
    )
    from crackseg.utils.checkpointing import CheckpointSaveConfig
    from crackseg.utils.logging import ExperimentLogger, get_logger
except ImportError:
    # Fallback for when running as standalone script
    create_dataloaders_from_config = None
    load_checkpoint = None
    save_checkpoint = None
    set_random_seeds = None
    CheckpointSaveConfig = None
    ExperimentLogger = None
    get_logger = None

# Explicitly define what this module exports
__all__ = [
    "create_dataloaders_from_config",
    "load_checkpoint",
    "save_checkpoint",
    "set_random_seeds",
    "CheckpointSaveConfig",
    "ExperimentLogger",
    "get_logger",
]
