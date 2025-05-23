"""
Script to generate and display a logical block diagram of the U-Net
architecture.

Loads the CNN U-Net model from config and uses the visualize_architecture
method to create a diagram (PNG) of the model structure.
"""

import os
import sys

import hydra

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, ".."))

from src.model.factory import create_unet  # noqa: E402


def main():
    """Load a CNN U-Net model from config and visualize its architecture."""
    # Initialize Hydra
    config_path = os.path.join(os.getcwd(), "configs")
    hydra.initialize_config_dir(config_dir=config_path, version_base=None)

    # Load the CNN U-Net config
    cfg = hydra.compose(config_name="model/unet_cnn")
    hydra.core.global_hydra.GlobalHydra.instance().clear()

    # Convert config to dictionary
    unet = create_unet(cfg.model)

    # Visualize architecture (will save as PNG and open it)
    print("Generando diagrama de arquitectura U-Net...")
    unet.visualize_architecture(filename="unet_architecture", view=True)
    print("Diagrama guardado como 'unet_architecture.png'.")


if __name__ == "__main__":
    main()
