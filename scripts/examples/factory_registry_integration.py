#!/usr/bin/env python
"""
Factory and Registry Integration Example.

This script demonstrates how to properly use the Factory and Registry system
to register and instantiate model components. It showcases the integration
between registry.py and factory.py for developing custom segmentation models.

Usage:
    python -m scripts.examples.factory_registry_integration

"""

import logging
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.model.registry_setup import (
    register_component,
    encoder_registry,
    bottleneck_registry,
    decoder_registry,
    architecture_registry
)
from src.model.factory import (
    create_unet
)
from src.model.base import (
    EncoderBase,
    BottleneckBase,
    DecoderBase,
    UNetBase
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


#
# STEP 1: Create custom components for demonstration
#
class SimpleEncoder(EncoderBase):
    """Simple encoder for demonstration purposes."""

    def __init__(self, in_channels, num_filters=16, **kwargs):
        super().__init__(in_channels=in_channels)
        self.conv = nn.Conv2d(in_channels, num_filters, kernel_size=3,
                              padding=1)
        self.relu = nn.ReLU(inplace=True)
        self._out_channels = num_filters
        self._skip_channels = [num_filters]

    def forward(self, x):
        features = self.relu(self.conv(x))
        return features, [features]  # Output and skip connections

    @property
    def out_channels(self):
        return self._out_channels

    @property
    def skip_channels(self):
        return self._skip_channels


class SimpleBottleneck(BottleneckBase):
    """Simple bottleneck for demonstration purposes."""

    def __init__(self, in_channels, expansion=2, **kwargs):
        super().__init__(in_channels=in_channels)
        self.conv = nn.Conv2d(
            in_channels, in_channels * expansion, kernel_size=3, padding=1
        )
        self.relu = nn.ReLU(inplace=True)
        self._out_channels = in_channels * expansion

    def forward(self, x):
        return self.relu(self.conv(x))

    @property
    def out_channels(self):
        return self._out_channels


class SimpleDecoder(DecoderBase):
    """Simple decoder for demonstration purposes."""

    def __init__(self, in_channels, skip_channels, out_channels=1, **kwargs):
        super().__init__(in_channels=in_channels, skip_channels=skip_channels)
        # Calculate total input channels (bottleneck + all skip connections)
        total_channels = in_channels + sum(skip_channels)

        self.conv = nn.Conv2d(
            total_channels, out_channels, kernel_size=3, padding=1
        )
        self._out_channels = out_channels

    def forward(self, x, skip_connections=None):
        # Handle the case where skip connections are None
        if skip_connections is None:
            skip_connections = []

        # Concatenate features from bottleneck and skip connections
        features = [x] + skip_connections
        features = torch.cat(features, dim=1)

        # Apply final convolution
        output = self.conv(features)
        return output

    @property
    def out_channels(self):
        return self._out_channels


class SimpleUNet(UNetBase):
    """Simple UNet for demonstration purposes."""

    def __init__(self, encoder, bottleneck, decoder):
        super().__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder

    def forward(self, x):
        # Encode
        encoded_features, skip_connections = self.encoder(x)

        # Apply bottleneck
        bottleneck_features = self.bottleneck(encoded_features)

        # Decode
        output = self.decoder(bottleneck_features, skip_connections)

        return output


#
# STEP 2: Register components
#
def register_demo_components():
    """Register components for the demonstration."""
    # Register encoder
    encoder_registry.register(
        name="SimpleEncoder", tags=["demo", "simple"]
    )(SimpleEncoder)
    log.info("Registered SimpleEncoder in encoder_registry")

    # Register bottleneck
    bottleneck_registry.register(
        name="SimpleBottleneck", tags=["demo", "simple"]
    )(SimpleBottleneck)
    log.info("Registered SimpleBottleneck in bottleneck_registry")

    # Register decoder
    decoder_registry.register(
        name="SimpleDecoder", tags=["demo", "simple"]
    )(SimpleDecoder)
    log.info("Registered SimpleDecoder in decoder_registry")

    # Register architecture
    architecture_registry.register(
        name="SimpleUNet", tags=["demo", "simple"]
    )(SimpleUNet)
    log.info("Registered SimpleUNet in architecture_registry")

    # Alternative registration using the register_component function
    @register_component("encoder", name="SimpleEncoder2", tags=["demo", "v2"])
    class SimpleEncoder2(SimpleEncoder):
        """A second version of SimpleEncoder for demonstration."""
        pass

    log.info("Registered SimpleEncoder2 using register_component")


#
# STEP 3: Create configurations
#
def create_demo_configs():
    """Create configuration dictionaries for the demonstration."""
    # Create component configurations
    encoder_config = {
        "type": "SimpleEncoder",
        "in_channels": 3,
        "num_filters": 32
    }

    bottleneck_config = {
        "type": "SimpleBottleneck",
        "in_channels": 32,  # This must match encoder output channels
        "expansion": 2
    }

    decoder_config = {
        "type": "SimpleDecoder",
        "in_channels": 64,  # This must match bottleneck output channels
        "skip_channels": [32],  # This must match encoder skip channels
        "out_channels": 1
    }

    # Create a UNet configuration
    unet_config = {
        "_target_": "src.model.unet.BaseUNet",
        "encoder": encoder_config,
        "bottleneck": bottleneck_config,
        "decoder": decoder_config
    }

    # Convert to OmegaConf for Hydra compatibility
    unet_config = OmegaConf.create(unet_config)

    return unet_config


#
# STEP 4: Use the factory to create components
#
def create_demo_models(config):
    """Create model instances using the factory."""
    log.info("Creating UNet model using factory...")

    # Create a basic UNet model
    basic_unet = create_unet(config)
    log.info(f"Created basic UNet model: {type(basic_unet).__name__}")

    # Create a UNet model with CBAM post-processing
    log.info("Creating UNet model with CBAM...")
    config_with_cbam = OmegaConf.create(OmegaConf.to_container(config))
    config_with_cbam.cbam_enabled = True
    config_with_cbam.cbam_params = {
        "reduction": 8,
        "kernel_size": 7
    }

    cbam_unet = create_unet(config_with_cbam)
    log.info(
        f"Created UNet model with CBAM: {type(cbam_unet).__name__}"
    )

    return basic_unet, cbam_unet


def main():
    """
    Run the factory and registry integration example.

    This demonstrates how to:
    1. Register custom components
    2. Create configurations
    3. Use the factory to instantiate models
    4. Apply post-processing like CBAM
    """
    log.info("Starting Factory and Registry Integration Example")

    # Register components
    register_demo_components()

    # List registered components
    log.info(f"Encoders: {encoder_registry.list()}")
    log.info(f"Bottlenecks: {bottleneck_registry.list()}")
    log.info(f"Decoders: {decoder_registry.list()}")
    log.info(f"Architectures: {architecture_registry.list()}")

    # Create configurations
    config = create_demo_configs()
    log.info(f"Created configuration: {OmegaConf.to_yaml(config)}")

    # Create models
    basic_unet, cbam_unet = create_demo_models(config)

    # Test models with a dummy input
    log.info("Testing models with dummy input...")
    x = torch.randn(1, 3, 32, 32)  # Batch, Channels, Height, Width

    # Run inference
    with torch.no_grad():
        basic_output = basic_unet(x)
        cbam_output = cbam_unet(x)

    log.info(f"Basic UNet output shape: {basic_output.shape}")
    log.info(f"CBAM UNet output shape: {cbam_output.shape}")

    log.info("Factory and Registry Integration Example completed!")


if __name__ == "__main__":
    main()
