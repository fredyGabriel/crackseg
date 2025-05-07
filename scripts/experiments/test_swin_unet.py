"""
Test script to demonstrate the U-Net with Swin Transformer V2 encoder.

This script instantiates a complete U-Net model using the Swin Transformer V2
encoder and tests it with a synthetic image.
"""

import os
import sys
from pathlib import Path
import torch
import hydra
import logging
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent.absolute()
os.chdir(project_root)
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def display_feature_maps(features, title=None):
    """Display feature maps from the model for visualization."""
    fig, axes = plt.subplots(1, len(features), figsize=(15, 3))
    if len(features) == 1:
        axes = [axes]

    for i, feature_map in enumerate(features):
        # Take the first feature map of the first batch
        feature = feature_map[0]
        # Sum across channels for visualization
        feature_viz = feature.sum(dim=0).detach().cpu().numpy()
        # Normalize for better visualization
        feature_viz = (feature_viz - feature_viz.min()) / (
            feature_viz.max() - feature_viz.min() + 1e-8
        )

        axes[i].imshow(feature_viz, cmap='viridis')
        axes[i].set_title(f"Stage {i+1}\n{feature.shape[0]} channels")
        axes[i].axis('off')

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def load_config(config_path):
    """Load configuration using Hydra."""
    # Load the configuration file directly
    conf_path = os.path.join(project_root, "configs", config_path)
    if not os.path.exists(conf_path):
        logger.error(f"Config file not found: {conf_path}")
        raise FileNotFoundError(f"Config file not found: {conf_path}")

    config = OmegaConf.load(conf_path)
    logger.info(f"Loaded configuration from {conf_path}")
    return config


def run_unet_swin(config_path="model/unet_swin.yaml"):
    """Run a test of U-Net with Swin Transformer encoder."""
    # Load configuration
    config = load_config(config_path)

    # Create a synthetic image for testing
    batch_size = 1
    channels = config.encoder.in_channels
    img_size = config.encoder.img_size
    x = torch.randn(batch_size, channels, img_size, img_size)

    logger.info(f"Created synthetic image with shape: {x.shape}")

    # Create the model
    model = hydra.utils.instantiate(config)
    logger.info(f"Created model: {type(model).__name__}")

    # Set model to evaluation mode
    model.eval()

    # Forward pass
    with torch.no_grad():
        output = model(x)

    logger.info(f"Model output shape: {output.shape}")

    # Access encoder features if available
    if hasattr(model, 'encoder_features'):
        encoder_features = model.encoder_features
        # Display features
        display_feature_maps(
            encoder_features,
            "Swin Transformer Encoder Features"
        )
    else:
        logger.warning("Model does not have encoder_features attribute.")

    # Display output
    plt.figure(figsize=(8, 8))
    plt.imshow(output[0, 0].detach().cpu().numpy(), cmap='gray')
    plt.title("Model Output")
    plt.axis('off')
    plt.colorbar()
    plt.show()

    return model, output


if __name__ == "__main__":
    run_unet_swin()
