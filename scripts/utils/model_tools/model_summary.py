"""
Script to display a summary of the U-Net model architecture.

This script loads a U-Net model from configuration and displays
a detailed summary, including layer structure, parameter counts,
and memory usage estimates.
"""

import os
import sys
from typing import Any, cast

import hydra
from hydra.core.global_hydra import GlobalHydra

# Add the project root directory to the Python path
sys.path.append(os.getcwd())

# Import project modules after adding to path
from crackseg.model.factory import create_unet  # noqa
from crackseg.model.base.abstract import UNetBase  # noqa


def main() -> None:
    """Load a model from config and print its summary."""
    # Initialize Hydra
    config_path = os.path.join(os.getcwd(), "configs")
    hydra.initialize_config_dir(config_dir=config_path, version_base=None)

    # Load the CNN U-Net config
    cfg = hydra.compose(config_name="model/unet_cnn")

    # Clean up hydra state
    GlobalHydra.instance().clear()

    # Create the model and cast to correct type
    model = create_unet(cfg.model)
    unet_model = cast(UNetBase, model)

    # Print model summary with example input shape
    unet_model.print_summary(input_shape=(1, 3, 512, 512))  # type: ignore

    # You can also get the summary as a dictionary for programmatic access
    summary_dict: dict[str, Any] = unet_model.summary(
        input_shape=(1, 3, 512, 512)
    )  # type: ignore

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
