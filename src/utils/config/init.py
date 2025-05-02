from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from typing import Optional
import os


def initialize_hydra(
    config_path: str = "../configs", job_name: str = "crackseg"
) -> None:
    """Initialize Hydra with the given config path (must be relative)."""
    # Get the absolute path to the config directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.normpath(os.path.join(current_dir, config_path))
    
    # Convert to relative path from CWD
    cwd = os.getcwd()
    config_dir = os.path.relpath(config_dir, cwd)
    
    initialize(config_path=config_dir, job_name=job_name, version_base=None)


def load_config(overrides: Optional[list] = None) -> DictConfig:
    """Load and compose configuration using Hydra. Optionally apply overrides.
    """
    if overrides is None:
        overrides = []
    cfg = compose(config_name="config", overrides=overrides)
    return cfg


def get_config(overrides: Optional[list] = None) -> DictConfig:
    """Get the complete configuration object."""
    initialize_hydra()
    cfg = load_config(overrides)
    return cfg


def print_config(cfg: DictConfig) -> None:
    """Print the configuration in a readable format."""
    print(
        OmegaConf.to_yaml(cfg)
    )
