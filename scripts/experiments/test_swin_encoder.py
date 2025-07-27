"""
Visual demonstration script for the Swin Transformer V2 encoder.

This script is a DEMONSTRATION tool that visualizes the operation of the
SwinTransformerEncoder with different input sizes and configurations.
Unlike unit tests, this script provides a visual and interactive representation
of the encoder, useful for presentations, research, and understanding the
model.
"""

import logging
import os
import sys
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from crackseg.model.encoder.swin_transformer_encoder import (
    SwinTransformerEncoder,
)

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent.absolute()
os.chdir(project_root)
sys.path.insert(0, str(project_root))

# Configure the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURE_MAPS_EXPECTED_DIM = 4


def visualize_feature_maps(feature_maps, title):
    """Visualizes selected feature maps from a tensor."""
    # Select only a few channels for visualization
    if feature_maps.dim() != FEATURE_MAPS_EXPECTED_DIM:
        logger.error(f"Invalid tensor shape: {feature_maps.shape}")
        return

    # Select up to 4 channels to visualize
    num_channels = min(4, feature_maps.size(1))
    _, axes = plt.subplots(1, num_channels, figsize=(15, 4))

    if num_channels == 1:
        axes = [axes]  # Convert to list if there's only one channel

    for i in range(num_channels):
        # Select a channel and convert it to numpy
        channel = feature_maps[0, i].detach().cpu().numpy()

        # Normalize for visualization
        channel = (channel - np.min(channel)) / (
            np.max(channel) - np.min(channel) + 1e-8
        )

        # Display in the subplot
        axes[i].imshow(channel, cmap="viridis")
        axes[i].set_title(f"Channel {i}")
        axes[i].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig):
    """Visual demonstration of the Swin Transformer V2 encoder."""
    # Load the specific encoder configuration
    encoder_cfg_path = os.path.join(
        project_root, "configs/model/encoder/swin_transformer_encoder.yaml"
    )
    encoder_cfg = OmegaConf.load(encoder_cfg_path)
    logger.info("Encoder configuration:")
    logger.info(OmegaConf.to_yaml(encoder_cfg))
    # Instantiate the encoder with the configuration file parameters
    encoder_params = OmegaConf.to_container(encoder_cfg, resolve=True)
    if not isinstance(encoder_params, dict):
        raise ValueError("Encoder configuration must be a dictionary")
    encoder_params.pop("in_channels", None)
    encoder = SwinTransformerEncoder(
        in_channels=encoder_cfg.in_channels,
        **dict(encoder_params),  # type: ignore[misc]
    )
    logger.info(f"Encoder created: {type(encoder).__name__}")

    # Model information
    logger.info("\n--- Model Feature Information ---")
    feature_info = encoder.get_feature_info()
    for info in feature_info:
        logger.info(
            f"Stage {info['stage']}: channels={info['channels']}, "
            f"reduction_factor={info['reduction_factor']}"
        )

    # DEMO 1: Feature visualization with different resolutions
    input_resolutions = [
        (256, 256),  # Base resolution
        (384, 256),  # Rectangular
        (128, 128),  # Low resolution
    ]

    logger.info(
        "\n=== DEMO: Feature visualization with different resolutions ==="
    )

    for res in input_resolutions:
        # Create test image (using noise for simplicity)
        input_tensor = torch.randn(1, 3, res[0], res[1])
        logger.info(f"\nProcessing image of {res[0]}x{res[1]} pixels")

        # Run encoder and get features
        with torch.no_grad():
            bottleneck, skip_connections = encoder(input_tensor)

        # Visualize low-level features (first skip connection)
        if len(skip_connections) > 0:
            visualize_feature_maps(
                skip_connections[0],
                f"Low-level features - Resolution {res[0]}x{res[1]}",
            )

        # Visualize bottleneck features
        visualize_feature_maps(
            bottleneck, f"Bottleneck features - Resolution {res[0]}x{res[1]}"
        )

    # DEMO 2: Comparison of input size handling methods
    non_standard_size = (225, 225)  # Size not divisible by patch_size
    input_tensor = torch.randn(
        1, 3, non_standard_size[0], non_standard_size[1]
    )

    logger.info("\n=== DEMO: Comparison of input size handling methods ===")
    logger.info(f"Using non-standard image size: {non_standard_size}")

    handling_methods = ["resize", "pad"]
    for method in handling_methods:
        logger.info(f"\nUsing method: {method}")

        # Configure the method
        encoder.handle_input_size = method

        # Process the image
        with torch.no_grad():
            try:
                bottleneck, skip_connections = encoder(input_tensor)

                # Visualize the bottleneck
                title = (
                    f"Bottleneck with '{method}' method - "
                    f"Input {non_standard_size}"
                )
                visualize_feature_maps(bottleneck, title)

                logger.info(f"Bottleneck shape: {bottleneck.shape}")
                logger.info("Skip connection shapes:")
                for i, skip in enumerate(skip_connections):
                    logger.info(f"  Level {i + 1}: {skip.shape}")

            except Exception as e:
                logger.error(f"Error with '{method}' method: {str(e)}")

    logger.info("\nDemonstration completed.")


if __name__ == "__main__":
    main()
