# pyright: reportPrivateUsage=false
"""
Tests for the hybrid architecture registry system.

Verifies that hybrid architectures can be registered, queried, and validated
correctly with different component combinations.
"""

from typing import Any

import pytest
import torch
from torch import nn

from crackseg.model import BottleneckBase, DecoderBase, EncoderBase
from crackseg.model.factory.hybrid_registry import (
    ComponentReference,
    HybridArchitectureDescriptor,
    hybrid_registry,
    query_architectures_by_component,
    query_architectures_by_tag,
    register_complex_hybrid,
    register_standard_hybrid,
)  # Migración: import actualizado para reflejar la ubicación real
from crackseg.model.factory.registry_setup import (
    architecture_registry,
    bottleneck_registry,
    decoder_registry,
    encoder_registry,
)


# Mock classes for testing
class MockEncoder(EncoderBase):
    def __init__(self, in_channels: int = 3):
        super().__init__(in_channels)
        self._out_channels = 8
        self._skip_channels = [4, 8]

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        output = torch.randn(1, self._out_channels, 8, 8)
        skips = [torch.randn(1, c, 16, 16) for c in self._skip_channels]
        return output, skips

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

        # Add bottleneck info (final output)
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
    def __init__(self, in_channels: int = 8):
        super().__init__(in_channels)
        self._out_channels = 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.randn(1, self._out_channels, 8, 8)

    @property
    def out_channels(self) -> int:
        return self._out_channels


class MockDecoder(DecoderBase):
    """
    MockDecoder que espera skip_channels en orden inverso a MockEncoder.
    Si MockEncoder.skip_channels = [16, 32],
    entonces MockDecoder.skip_channels = [32, 16].
    """

    def __init__(
        self, in_channels: int = 64, skip_channels: list[int] | None = None
    ):
        # Si no se pasa skip_channels, usar el reverso del default de
        # MockEncoder
        if skip_channels is None:
            # reverse de [16, 32]
            skip_channels = [32, 16]
        super().__init__(in_channels, skip_channels=skip_channels)
        self._out_channels = 1

    def forward(
        self, x: torch.Tensor, skips: list[torch.Tensor]
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        return torch.randn(
            batch_size, self._out_channels, x.shape[2] * 2, x.shape[3] * 2
        )

    @property
    def out_channels(self) -> int:
        return self._out_channels


class MockAttention(nn.Module):
    def __init__(self):
        super().__init__()


def setup_module(module: object) -> None:
    """Set up test module - register test components."""
    # Unregister test components if already present
    try:
        encoder_registry.unregister("MockEncoder")
    except Exception:
        pass
    try:
        bottleneck_registry.unregister("MockBottleneck")
    except Exception:
        pass
    try:
        decoder_registry.unregister("MockDecoder")
    except Exception:
        pass
    try:
        from crackseg.model.factory.registry_setup import (
            component_registries,
        )

        attention_registry = component_registries.get("attention")
        if attention_registry is not None:
            attention_registry.unregister("MockAttention")
    except Exception:
        pass
    # Register test components in appropriate registries
    encoder_registry.register(name="MockEncoder")(MockEncoder)
    bottleneck_registry.register(name="MockBottleneck")(MockBottleneck)
    decoder_registry.register(name="MockDecoder")(MockDecoder)
    # Get attention registry and register mock component
    from crackseg.model.factory.registry_setup import component_registries

    attention_registry = component_registries.get("attention")
    if attention_registry is not None:
        attention_registry.register(name="MockAttention")(MockAttention)


def teardown_module(module: object) -> None:
    """Clean up after tests."""
    # Clear hybrid registry
    hybrid_registry._descriptors = {}

    # Unregister test components to avoid duplicate registration errors
    try:
        encoder_registry.unregister("MockEncoder")
    except Exception:
        pass
    try:
        bottleneck_registry.unregister("MockBottleneck")
    except Exception:
        pass
    try:
        decoder_registry.unregister("MockDecoder")
    except Exception:
        pass
    try:
        from crackseg.model.factory.registry_setup import (
            component_registries,
        )

        attention_registry = component_registries.get("attention")
        if attention_registry is not None:
            attention_registry.unregister("MockAttention")
    except Exception:
        pass


def test_component_reference_validation():
    """Test that component references are validated correctly."""
    # Valid component
    valid_ref = ComponentReference("encoder", "MockEncoder")
    assert valid_ref.validate() is True

    # Invalid component - should raise ValueError
    invalid_ref = ComponentReference("encoder", "NonExistentEncoder")
    with pytest.raises(ValueError):
        invalid_ref.validate()

    # Optional component that doesn't exist - should warn but not raise
    optional_ref = ComponentReference(
        "encoder", "NonExistentEncoder", optional=True
    )
    assert optional_ref.validate() is False


def test_hybrid_architecture_descriptor():
    """Test hybrid architecture descriptor creation and validation."""
    components = {
        "encoder": ComponentReference("encoder", "MockEncoder"),
        "bottleneck": ComponentReference("bottleneck", "MockBottleneck"),
        "decoder": ComponentReference("decoder", "MockDecoder"),
    }

    # Create descriptor
    descriptor = HybridArchitectureDescriptor(
        name="TestArchitecture", components=components, tags=["test", "mock"]
    )

    # Validate the descriptor
    assert descriptor.validate() is True

    # Test tag list generation
    tag_list = descriptor.to_tag_list()
    assert "encoder:MockEncoder" in tag_list
    assert "bottleneck:MockBottleneck" in tag_list
    assert "decoder:MockDecoder" in tag_list
    assert "test" in tag_list
    assert "mock" in tag_list


def test_register_standard_hybrid():
    """Test registering a standard hybrid architecture."""
    # Register a standard hybrid
    result = register_standard_hybrid(
        name="StandardHybrid",
        encoder_type="MockEncoder",
        bottleneck_type="MockBottleneck",
        decoder_type="MockDecoder",
    )

    assert result is True
    descriptors = hybrid_registry._descriptors
    assert "StandardHybrid" in descriptors

    # Register with attention
    result = register_standard_hybrid(
        name="StandardHybridWithAttention",
        encoder_type="MockEncoder",
        bottleneck_type="MockBottleneck",
        decoder_type="MockDecoder",
        attention_type="MockAttention",
        tags=["attention"],
    )

    assert result is True
    descriptors = hybrid_registry._descriptors
    assert "StandardHybridWithAttention" in descriptors

    # Try to register duplicate - should raise ValueError
    with pytest.raises(ValueError):
        register_standard_hybrid(
            name="StandardHybrid",
            encoder_type="MockEncoder",
            bottleneck_type="MockBottleneck",
            decoder_type="MockDecoder",
        )

    # Try to register with non-existent component - should raise ValueError
    with pytest.raises(ValueError):
        register_standard_hybrid(
            name="InvalidHybrid",
            encoder_type="NonExistentEncoder",
            bottleneck_type="MockBottleneck",
            decoder_type="MockDecoder",
        )


def test_register_complex_hybrid():
    """Test registering a complex hybrid architecture."""
    # Define components
    components = {
        "spatial_encoder": ("encoder", "MockEncoder"),
        "temporal_bottleneck": ("bottleneck", "MockBottleneck"),
        "spatial_decoder": ("decoder", "MockDecoder"),
        "attention_module": ("attention", "MockAttention"),
    }

    # Register complex hybrid
    result = register_complex_hybrid(
        name="ComplexHybrid",
        components=components,
        tags=["complex", "spatial-temporal"],
    )

    assert result is True
    descriptors = hybrid_registry._descriptors
    assert "ComplexHybrid" in descriptors

    # Get the descriptor to verify
    descriptor = hybrid_registry.get_descriptor("ComplexHybrid")
    assert len(descriptor.components) == 4  # noqa: PLR2004
    assert "spatial_encoder" in descriptor.components
    assert "temporal_bottleneck" in descriptor.components
    assert "attention_module" in descriptor.components
    assert "complex" in descriptor.tags


def test_query_architectures():
    """Test querying architectures by component and tag."""
    # Should find at least the standard and complex hybrids
    encoder_architectures = query_architectures_by_component("MockEncoder")
    assert "StandardHybrid" in encoder_architectures
    assert "ComplexHybrid" in encoder_architectures

    # Test query by role
    spatial_encoder_architectures = query_architectures_by_component(
        "MockEncoder", role="spatial_encoder"
    )
    assert "ComplexHybrid" in spatial_encoder_architectures
    assert "StandardHybrid" not in spatial_encoder_architectures

    # Test query by tag
    complex_architectures = query_architectures_by_tag("complex")
    assert "ComplexHybrid" in complex_architectures
    assert "StandardHybrid" not in complex_architectures

    attention_architectures = query_architectures_by_tag("attention")
    assert "StandardHybridWithAttention" in attention_architectures


def test_architecture_registry_integration():
    """Test that hybrid architectures are also in the main architecture
    registry."""

    def get_descriptors(
        registry: object,
    ) -> dict[str, HybridArchitectureDescriptor]:
        return registry._descriptors  # type: ignore[attr-defined]

    arch_registry_items = get_descriptors(architecture_registry)
    assert "StandardHybrid" in arch_registry_items
    assert "ComplexHybrid" in arch_registry_items
    assert "StandardHybridWithAttention" in arch_registry_items
