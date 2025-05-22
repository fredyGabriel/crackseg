"""
Script to display a summary of the U-Net model architecture.

This script loads a U-Net model from configuration and displays
a detailed summary, including layer structure, parameter counts,
and memory usage estimates.
"""

import os
import sys

import hydra
from omegaconf import OmegaConf

# Add the project root directory to the Python path
sys.path.append(os.getcwd())

# Import project modules after adding to path
from src.model.factory import create_unet  # noqa


def main():
    """Load a model from config and print its summary."""
    # Initialize Hydra
    config_path = os.path.join(os.getcwd(), "configs")
    hydra.initialize_config_dir(config_dir=config_path, version_base=None)

    # Load the CNN U-Net config
    cfg = hydra.compose(config_name="model/unet_cnn")

    # Clean up hydra state
    hydra.core.global_hydra.GlobalHydra.instance().clear()

    # Convert config to dictionary
    config_dict = OmegaConf.to_container(cfg.model, resolve=True)

    # Create the model
    model = create_unet(config_dict)

    # Print model summary with example input shape
    model.print_summary(input_shape=(1, 3, 512, 512))

    # You can also get the summary as a dictionary for programmatic access
    summary_dict = model.summary(input_shape=(1, 3, 512, 512))

    # Example: Extract specific information
    total_params = summary_dict["parameters"]["total"]
    memory_usage = summary_dict["memory_usage"]["total_estimated_mb"]

    print(
        f"\nIn summary: Model has {total_params:,} parameters and "
        f"requires approximately {memory_usage:.2f} MB of memory "
        f"for inference with a 512x512 input image."
    )


if __name__ == "__main__":
    main()
