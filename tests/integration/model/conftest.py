"""Configuration and fixtures for model integration tests."""

from collections.abc import Generator
from typing import Any

import pytest
import torch

# Import Base classes
from crackseg.model import BottleneckBase, DecoderBase, EncoderBase
from crackseg.model.bottleneck.cnn_bottleneck import BottleneckBlock
from crackseg.model.decoder.cnn_decoder import CNNDecoder

# Import CNN components
from crackseg.model.encoder.cnn_encoder import CNNEncoder

# Import Registries needed for fixture
from crackseg.model.factory.registry_setup import (
    bottleneck_registry,
    decoder_registry,
    encoder_registry,
)

# --- Mock Components for Manual Config Tests ---


class MockEncoder(EncoderBase):
    def __init__(self, in_channels: int):
        super().__init__(in_channels)
        self._out_channels = 64
        # Contract: skip_channels = [16, 32] (high->low resolution)
        self._skip_channels = [16, 32]

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # Return dummy output and list of skips
        batch_size = x.shape[0]
        output_features = torch.randn(
            batch_size, self._out_channels, x.shape[2] // 4, x.shape[3] // 4
        )
        skips = [
            torch.randn(
                batch_size,
                c,
                x.shape[2] // (2 ** (i + 1)),
                x.shape[3] // (2 ** (i + 1)),
            )
            for i, c in enumerate(self._skip_channels)
        ]
        return output_features, skips

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @property
    def skip_channels(self) -> list[int]:
        return self._skip_channels

    def get_feature_info(self) -> list[dict[str, Any]]:
        """
        Get information about feature maps produced by the mock encoder.

        Returns:
            List[Dict[str, Any]]: Information about each feature map,
                                 including channels and reduction factor.
        """
        feature_info: list[dict[str, Any]] = []

        # Mock encoder with 2 stages: reduction factors 2 and 4
        reduction_factors = [2, 4]

        for i, channels in enumerate(self._skip_channels):
            feature_info.append(
                {
                    "channels": channels,
                    "reduction": reduction_factors[i],
                    "stage": i,
                }
            )

        # Add bottleneck info (final output) with reduction factor 4
        feature_info.append(
            {
                "channels": self._out_channels,
                "reduction": 4,
                "stage": len(self._skip_channels),
            }
        )

        return feature_info

    @property
    def feature_info(self) -> list[dict[str, Any]]:
        """Information about output features for each stage.

        Returns:
            List of dictionaries, each containing:
                - 'channels': Number of output channels
                - 'reduction': Spatial reduction factor from input
                - 'stage': Stage index
        """
        return self.get_feature_info()


class MockBottleneck(BottleneckBase):
    def __init__(self, in_channels: int):
        super().__init__(in_channels)
        self._out_channels = 128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        return torch.randn(
            batch_size, self._out_channels, x.shape[2], x.shape[3]
        )

    @property
    def out_channels(self) -> int:
        return self._out_channels


# Dummy for Identity Bottleneck
class DummyIdentity(BottleneckBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        **kwargs: object,
    ):
        super().__init__(in_channels=in_channels)
        self.in_channels = in_channels
        self._out_channels: int = (
            out_channels if out_channels is not None else in_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.in_channels != self._out_channels:
            # Simple way to change channels if needed for testing
            return torch.randn(
                x.shape[0], self._out_channels, x.shape[2], x.shape[3]
            )
        return x

    @property
    def out_channels(self) -> int:
        return self._out_channels


class TestDecoderImpl(DecoderBase):
    def __init__(self, in_channels: int, skip_channels: list[int]):
        # Contract: skip_channels debe ser el reverse de
        # MockEncoder.skip_channels, i.e., [32, 16] (low->high resolution)
        super().__init__(in_channels, skip_channels)
        self._out_channels: int = 1

    def forward(
        self, x: torch.Tensor, skips: list[torch.Tensor]
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        return torch.randn(
            batch_size, self._out_channels, x.shape[2] * 4, x.shape[3] * 4
        )

    @property
    def out_channels(self) -> int:
        return self._out_channels


# --- End Mock Components ---


# --- Fixtures ---


@pytest.fixture(scope="function")
def register_mock_components() -> Generator[None, None, None]:  # noqa: PLR0912
    """Temporarily registers mock components for a test function."""
    registered_names: dict[str, list[str]] = {
        "encoder": [],
        "bottleneck": [],
        "decoder": [],
    }
    try:
        # Register Encoder if not present
        if "MockEncoder" not in encoder_registry:
            encoder_registry.register(name="MockEncoder")(MockEncoder)
            registered_names["encoder"].append("MockEncoder")
        # Register Bottleneck if not present
        if "MockBottleneck" not in bottleneck_registry:
            bottleneck_registry.register(name="MockBottleneck")(MockBottleneck)
            registered_names["bottleneck"].append("MockBottleneck")
        # Register Decoder if not present
        if "TestDecoderImpl" not in decoder_registry:
            decoder_registry.register(name="TestDecoderImpl")(TestDecoderImpl)
            registered_names["decoder"].append("TestDecoderImpl")

        # Registro explícito de componentes CNN reales para integración
        if "CNNEncoder" not in encoder_registry:
            encoder_registry.register(name="CNNEncoder")(CNNEncoder)
        if "BottleneckBlock" not in bottleneck_registry:
            bottleneck_registry.register(name="BottleneckBlock")(
                BottleneckBlock
            )
        if "CNNDecoder" not in decoder_registry:
            decoder_registry.register(name="CNNDecoder")(CNNDecoder)

        yield  # Test runs here

    finally:
        # Teardown: Unregister only the components registered by this fixture
        # Use try/except for robustness when unregistering
        for name in registered_names["encoder"]:
            try:
                encoder_registry.unregister(name)
            except KeyError:
                pass  # Ignore if already unregistered
        for name in registered_names["bottleneck"]:
            try:
                bottleneck_registry.unregister(name)
            except KeyError:
                pass  # Ignore if already unregistered
        for name in registered_names["decoder"]:
            try:
                decoder_registry.unregister(name)
            except KeyError:
                pass  # Ignore if already unregistered


def pytest_configure(config: object) -> None:
    # ... registro de mocks ...
    # Registro explícito de componentes CNN reales para integración
    if "CNNEncoder" not in encoder_registry:
        encoder_registry.register(name="CNNEncoder")(CNNEncoder)
    if "BottleneckBlock" not in bottleneck_registry:
        bottleneck_registry.register(name="BottleneckBlock")(BottleneckBlock)
    if "CNNDecoder" not in decoder_registry:
        decoder_registry.register(name="CNNDecoder")(CNNDecoder)
