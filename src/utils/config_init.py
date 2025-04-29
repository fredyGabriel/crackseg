from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from typing import Optional


def initialize_hydra(
    config_path: str = "configs", job_name: str = "crackseg"
) -> None:
    """Initialize Hydra with the given config path (must be relative)."""
    initialize(config_path=config_path, job_name=job_name, version_base=None)


def load_config(overrides: Optional[list] = None) -> DictConfig:
    """Load and compose configuration using Hydra. Optionally apply overrides.
    """
    if overrides is None:
        overrides = []
    cfg = compose(config_name="base", overrides=overrides)
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
