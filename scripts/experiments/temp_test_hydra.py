"""Temporary script to test Hydra config loading and model instantiation."""

import os

# Adjust path if necessary to find the modules
import sys

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Ensure mock components are registered
from src.model.base import UNetBase
from src.model.factory import create_unet


@hydra.main(
    config_path="../configs", config_name="model/unet_mock", version_base=None
)
def run_test(cfg: DictConfig):
    """Load config, instantiate model, and run forward pass."""
    print("Configuration loaded successfully:")
    print(OmegaConf.to_yaml(cfg))

    try:
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        print("\nAttempting to create UNet model from model config...")
        unet = create_unet(config_dict["model"])
        print("UNet model created successfully!")
        assert isinstance(unet, UNetBase)

        print("\nChecking component types...")
        print(f"  Encoder type: {type(unet.encoder)}")
        print(f"  Bottleneck type: {type(unet.bottleneck)}")
        print(f"  Decoder type: {type(unet.decoder)}")

        print("\nRunning forward pass...")
        unet.eval()
        input_channels = unet.get_input_channels()
        x = torch.randn(2, input_channels, 64, 64)
        with torch.no_grad():
            output = unet(x)
        print(f"Forward pass completed. Output shape: {output.shape}")

        print("\nTest completed successfully!")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_test()
