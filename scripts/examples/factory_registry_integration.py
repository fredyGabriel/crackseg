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
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from crackseg.model.base import (
    BottleneckBase,
    DecoderBase,
    EncoderBase,
    UNetBase,
)
from crackseg.model.factory import create_unet
from crackseg.model.factory.registry_setup import (
    architecture_registry,
    bottleneck_registry,
    decoder_registry,
    encoder_registry,
    register_component,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)


#
# STEP 1: Create custom components for demonstration
#
class SimpleEncoder(EncoderBase):
    """Simple encoder for demonstration purposes."""

    def __init__(
        self, in_channels: int, num_filters: int = 16, **kwargs: Any
    ) -> None:
        super().__init__(in_channels=in_channels)
        self.conv = nn.Conv2d(
            in_channels, num_filters, kernel_size=3, padding=1
        )
        self.relu = nn.ReLU(inplace=True)
        self._out_channels: int = num_filters
        self._skip_channels: list[int] = [num_filters]

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        features = self.relu(self.conv(x))
        return features, [features]  # Output and skip connections

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @property
    def skip_channels(self) -> list[int]:
        return self._skip_channels


class SimpleBottleneck(BottleneckBase):
    """Simple bottleneck for demonstration purposes."""

    def __init__(
        self, in_channels: int, expansion: int = 2, **kwargs: Any
    ) -> None:
        super().__init__(in_channels=in_channels)
        self.conv = nn.Conv2d(
            in_channels, in_channels * expansion, kernel_size=3, padding=1
        )
        self.relu = nn.ReLU(inplace=True)
        self._out_channels: int = in_channels * expansion
        self._in_channels: int = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x))

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels


class SimpleDecoder(DecoderBase):
    """Simple decoder for demonstration purposes."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: list[int],
        out_channels: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(in_channels=in_channels, skip_channels=skip_channels)
        total_channels = in_channels + sum(skip_channels)
        self.conv = nn.Conv2d(
            total_channels, out_channels, kernel_size=3, padding=1
        )
        self._out_channels: int = out_channels

    def forward(
        self, x: torch.Tensor, skips: list[torch.Tensor]
    ) -> torch.Tensor:
        features = [x] + skips
        features_cat = torch.cat(features, dim=1)
        output = self.conv(features_cat)
        return output

    @property
    def out_channels(self) -> int:
        return self._out_channels


class SimpleUNet(UNetBase):
    """Simple UNet for demonstration purposes."""

    def __init__(
        self,
        encoder: EncoderBase,
        bottleneck: BottleneckBase,
        decoder: DecoderBase,
    ) -> None:
        super().__init__(encoder, bottleneck, decoder)
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Garanty that the components are not None (for linter and security)
        assert self.encoder is not None, "Encoder must not be None"
        assert self.bottleneck is not None, "Bottleneck must not be None"
        assert self.decoder is not None, "Decoder must not be None"
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
    encoder_registry.register(name="SimpleEncoder", tags=["demo", "simple"])(
        SimpleEncoder
    )
    log.info("Registered SimpleEncoder in encoder_registry")

    # Register bottleneck
    bottleneck_registry.register(
        name="SimpleBottleneck", tags=["demo", "simple"]
    )(SimpleBottleneck)
    log.info("Registered SimpleBottleneck in bottleneck_registry")

    # Register decoder
    decoder_registry.register(name="SimpleDecoder", tags=["demo", "simple"])(
        SimpleDecoder
    )
    log.info("Registered SimpleDecoder in decoder_registry")

    # Register architecture
    architecture_registry.register(name="SimpleUNet", tags=["demo", "simple"])(
        SimpleUNet
    )
    log.info("Registered SimpleUNet in architecture_registry")

    # The decorator does not have explicit typing, but it is safe in this
    # context because it only registers the class.
    @register_component("encoder", name="SimpleEncoder2", tags=["demo", "v2"])
    class SimpleEncoder2(SimpleEncoder):
        """A second version of SimpleEncoder for demonstration."""

        pass

    _ = SimpleEncoder2  # Mark as used for linter/static analysis


#
# STEP 3: Create configurations
#
def create_demo_configs():
    """Create configuration dictionaries for the demonstration."""
    # Create component configurations
    encoder_config = {
        "type": "SimpleEncoder",
        "in_channels": 3,
        "num_filters": 32,
    }

    bottleneck_config = {
        "type": "SimpleBottleneck",
        "in_channels": 32,  # This must match encoder output channels
        "expansion": 2,
    }

    decoder_config = {
        "type": "SimpleDecoder",
        "in_channels": 64,  # This must match bottleneck output channels
        "skip_channels": [32],  # This must match encoder skip channels
        "out_channels": 1,
    }

    # Create a UNet configuration
    unet_dict = {
        "_target_": "src.model.unet.BaseUNet",
        "encoder": encoder_config,
        "bottleneck": bottleneck_config,
        "decoder": decoder_config,
    }

    # Convert to OmegaConf for Hydra compatibility
    unet_config = OmegaConf.create(unet_dict)
    assert isinstance(unet_config, DictConfig)

    return unet_config


#
# STEP 4: Use the factory to create components
#
def create_demo_models(config: DictConfig) -> None:
    """
    Crea modelos de demostración a partir de una configuración DictConfig.
    """
    log.info("Creating UNet model using factory...")

    # Create a basic UNet model
    basic_unet = create_unet(config)
    log.info(f"Created basic UNet model: {type(basic_unet).__name__}")

    # Create a UNet model with CBAM post-processing
    log.info("Creating UNet model with CBAM...")
    config_with_cbam = OmegaConf.create(
        OmegaConf.to_container(config, resolve=True)
    )
    assert isinstance(config_with_cbam, DictConfig)
    config_with_cbam.cbam_enabled = True
    config_with_cbam.cbam_params = {"reduction": 8, "kernel_size": 7}

    cbam_unet = create_unet(config_with_cbam)
    log.info(f"Created UNet model with CBAM: {type(cbam_unet).__name__}")


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
    log.info(f"Encoders: {encoder_registry.list_components()}")
    log.info(f"Bottlenecks: {bottleneck_registry.list_components()}")
    log.info(f"Decoders: {decoder_registry.list_components()}")
    log.info(f"Architectures: {architecture_registry.list_components()}")

    # Create configurations
    config = create_demo_configs()
    log.info(f"Created configuration: {OmegaConf.to_yaml(config)}")

    # Create models
    create_demo_models(config)

    log.info("Factory and Registry Integration Example completed!")


if __name__ == "__main__":
    main()
