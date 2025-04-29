from omegaconf import DictConfig, OmegaConf
from typing import List, Optional


def apply_overrides(
    cfg: DictConfig, overrides: Optional[List[str]] = None
) -> DictConfig:
    """Apply a list of override strings to a configuration using Hydra syntax.
    """
    if overrides is None:
        overrides = []
    # Hydra compose can be used to apply overrides at load time
    # This function is a placeholder for custom logic if needed
    return OmegaConf.merge(
        cfg,
        OmegaConf.from_dotlist(overrides)
    )


def save_config(cfg: DictConfig, path: str) -> None:
    """Save the configuration to a YAML file."""
    with open(path, "w", encoding="utf-8") as f:
        OmegaConf.save(config=cfg, f=f.name)


def example_override_usage() -> None:
    """Example: how to use overrides with Hydra from the command line."""
    print("Command-line override example:")
    print(
        "python main.py "
        "training.epochs=100 "
        "model.model_name=deeplabv3+"
    )
    print(
        "This will override the values of epochs and "
        "model_name in the configuration."
    )

# Note: actual override integration is done when loading the config with Hydra,
# using the 'overrides' parameter in compose or from the CLI.
# These functions allow you to manipulate and save configurations
# programmatically if needed.
