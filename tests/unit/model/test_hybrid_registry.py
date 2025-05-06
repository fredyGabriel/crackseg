"""
Tests for the hybrid architecture registry system.

Verifies that hybrid architectures can be registered, queried, and validated
correctly with different component combinations.
"""

import pytest
import torch
import torch.nn as nn
from src.model.base import EncoderBase, BottleneckBase, DecoderBase

from src.model.registry_setup import (
    encoder_registry,
    bottleneck_registry,
    decoder_registry,
    architecture_registry
)
from src.model.hybrid_registry import (
    hybrid_registry,
    register_standard_hybrid,
    register_complex_hybrid,
    query_architectures_by_component,
    query_architectures_by_tag,
    HybridArchitectureDescriptor,
    ComponentReference
)


# Mock classes for testing
class MockEncoder(EncoderBase):
    def __init__(self, in_channels=3):
        super().__init__(in_channels)
        self._out_channels = 8
        self._skip_channels = [4, 8]

    def forward(self, x):
        output = torch.randn(1, self._out_channels, 8, 8)
        skips = [torch.randn(1, c, 16, 16) for c in self._skip_channels]
        return output, skips

    @property
    def out_channels(self):
        return self._out_channels

    @property
    def skip_channels(self):
        return self._skip_channels


class MockBottleneck(BottleneckBase):
    def __init__(self, in_channels=8):
        super().__init__(in_channels)
        self._out_channels = 8

    def forward(self, x):
        return torch.randn(1, self._out_channels, 8, 8)

    @property
    def out_channels(self):
        return self._out_channels


class MockDecoder(DecoderBase):
    def __init__(self, in_channels=8, skip_channels=[4, 8]):
        super().__init__(in_channels, skip_channels)
        self._out_channels = 1

    def forward(self, x, skips):
        batch_size = x.shape[0]
        return torch.randn(batch_size, self._out_channels, x.shape[2]*2,
                           x.shape[3]*2)

    @property
    def out_channels(self):
        return self._out_channels


class MockAttention(nn.Module):
    def __init__(self):
        super().__init__()


def setup_module(module):
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
        from src.model.registry_setup import component_registries
        attention_registry = component_registries.get('attention')
        attention_registry.unregister("MockAttention")
    except Exception:
        pass
    # Register test components in appropriate registries
    encoder_registry.register(name="MockEncoder")(MockEncoder)
    bottleneck_registry.register(name="MockBottleneck")(MockBottleneck)
    decoder_registry.register(name="MockDecoder")(MockDecoder)
    # Get attention registry and register mock component
    from src.model.registry_setup import component_registries
    attention_registry = component_registries.get('attention')
    attention_registry.register(name="MockAttention")(MockAttention)


def teardown_module(module):
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
        from src.model.registry_setup import component_registries
        attention_registry = component_registries.get('attention')
        attention_registry.unregister("MockAttention")
    except Exception:
        pass


def test_component_reference_validation():
    """Test that component references are validated correctly."""
    # Valid component
    valid_ref = ComponentReference('encoder', 'MockEncoder')
    assert valid_ref.validate() is True

    # Invalid component - should raise ValueError
    invalid_ref = ComponentReference('encoder', 'NonExistentEncoder')
    with pytest.raises(ValueError):
        invalid_ref.validate()

    # Optional component that doesn't exist - should warn but not raise
    optional_ref = ComponentReference(
        'encoder', 'NonExistentEncoder', optional=True
    )
    assert optional_ref.validate() is False


def test_hybrid_architecture_descriptor():
    """Test hybrid architecture descriptor creation and validation."""
    components = {
        'encoder': ComponentReference('encoder', 'MockEncoder'),
        'bottleneck': ComponentReference('bottleneck', 'MockBottleneck'),
        'decoder': ComponentReference('decoder', 'MockDecoder'),
    }

    # Create descriptor
    descriptor = HybridArchitectureDescriptor(
        name="TestArchitecture",
        components=components,
        tags=["test", "mock"]
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
        decoder_type="MockDecoder"
    )

    assert result is True
    assert "StandardHybrid" in hybrid_registry.list_architectures()

    # Register with attention
    result = register_standard_hybrid(
        name="StandardHybridWithAttention",
        encoder_type="MockEncoder",
        bottleneck_type="MockBottleneck",
        decoder_type="MockDecoder",
        attention_type="MockAttention",
        tags=["attention"]
    )

    assert result is True
    assert "StandardHybridWithAttention" in hybrid_registry.list_architectures(
    )

    # Try to register duplicate - should raise ValueError
    with pytest.raises(ValueError):
        register_standard_hybrid(
            name="StandardHybrid",
            encoder_type="MockEncoder",
            bottleneck_type="MockBottleneck",
            decoder_type="MockDecoder"
        )

    # Try to register with non-existent component - should raise ValueError
    with pytest.raises(ValueError):
        register_standard_hybrid(
            name="InvalidHybrid",
            encoder_type="NonExistentEncoder",
            bottleneck_type="MockBottleneck",
            decoder_type="MockDecoder"
        )


def test_register_complex_hybrid():
    """Test registering a complex hybrid architecture."""
    # Define components
    components = {
        'spatial_encoder': ('encoder', 'MockEncoder'),
        'temporal_bottleneck': ('bottleneck', 'MockBottleneck'),
        'spatial_decoder': ('decoder', 'MockDecoder'),
        'attention_module': ('attention', 'MockAttention'),
    }

    # Register complex hybrid
    result = register_complex_hybrid(
        name="ComplexHybrid",
        components=components,
        tags=["complex", "spatial-temporal"]
    )

    assert result is True
    assert "ComplexHybrid" in hybrid_registry.list_architectures()

    # Get the descriptor to verify
    descriptor = hybrid_registry.get_descriptor("ComplexHybrid")
    assert len(descriptor.components) == 4
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
    # Check that the standard and complex hybrids are in the architecture
    # registry
    arch_registry_items = architecture_registry.list()
    assert "StandardHybrid" in arch_registry_items
    assert "ComplexHybrid" in arch_registry_items
    assert "StandardHybridWithAttention" in arch_registry_items
