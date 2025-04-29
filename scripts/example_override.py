"""
Example script: How to use the configuration system with overrides

This script demonstrates how to create a base configuration, apply command-line
style overrides, and print the resulting configuration in YAML format. This is
useful for testing and for users who want to understand how to modify
configuration values dynamically.
"""

from omegaconf import OmegaConf
from src.utils.config_override import apply_overrides

# Base configuration (could be loaded from file or defined inline)
base_config = {
    'training': {'epochs': 10},
    'model': {'model_name': 'unet'}
}

# Create a DictConfig object from the base dictionary
cfg = OmegaConf.create(base_config)

# Example overrides (as would be passed from the command line)
overrides = [
    'training.epochs=99',
    'model.model_name=deeplabv3+'
]

# Apply the overrides to the configuration
cfg_overridden = apply_overrides(cfg, overrides)

# Print the final configuration in YAML format
print("Final configuration with overrides applied:")
print(OmegaConf.to_yaml(cfg_overridden))
