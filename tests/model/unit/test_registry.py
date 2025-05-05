"""Unit tests for the model component registry system."""

import pytest
from typing import List

from src.model.registry import Registry
from src.model.base import EncoderBase, BottleneckBase, DecoderBase


# Define simple mock classes for testing
class MockEncoder(EncoderBase):
    """Mock implementation of EncoderBase for testing."""
    def __init__(self, in_channels: int = 3):
        super().__init__(in_channels)

    def forward(self, x):
        return x, []

    @property
    def out_channels(self) -> int:
        return 64

    @property
    def skip_channels(self) -> List[int]:
        return [16, 32]


class MockBottleneck(BottleneckBase):
    """Mock implementation of BottleneckBase for testing."""
    def __init__(self, in_channels: int = 64):
        super().__init__(in_channels)

    def forward(self, x):
        return x

    @property
    def out_channels(self) -> int:
        return 128


class MockDecoder(DecoderBase):
    """Mock implementation of DecoderBase for testing."""
    def __init__(
        self,
        in_channels: int = 128,
        skip_channels: list = None,
        skip_channels_list: list = None
    ):
        # Permitir ambos nombres de argumento para compatibilidad Hydra
        if skip_channels_list is not None:
            skips_provided = skip_channels_list
        elif skip_channels is not None:
            skips_provided = skip_channels
        else:
            # Default: skip_channels invertidos respecto a MockEncoder
            # MockEncoder.skip_channels = [16, 32]
            # Por lo tanto, usamos [32, 16] aquí (de menor a mayor resolución)
            skips_provided = [32, 16]

        # IMPORTANT: In UNet architecture, decoder.skip_channels must be
        # the reverse of encoder.skip_channels, since the decoder processes
        # features from low to high resolution while the encoder outputs
        # them from high to low resolution.
        super().__init__(in_channels, skips_provided)

    def forward(self, x, skips):
        # Note: skips passed here by UNet will be in encoder order
        # (high to low res). If this mock needed to *use* skips,
        # it might need to reverse them.
        return x

    @property
    def out_channels(self) -> int:
        return 1

    @classmethod
    def skip_channels_expected(cls):
        """
        Helper for tests: Return the expected skip_channels list after
        reversal.

        This matches the order expected for decoder.skip_channels when
        connected to a MockEncoder (low-res to high-res).
        """
        # [32, 16] is the reverse of MockEncoder.skip_channels [16, 32]
        return [32, 16]


# Fixtures for common test setups
@pytest.fixture
def encoder_registry():
    """Create a new EncoderBase registry for testing."""
    return Registry(EncoderBase, "Encoder")


@pytest.fixture
def populated_registry(encoder_registry):
    """Create a registry with some pre-registered components."""
    @encoder_registry.register()
    class TestEncoder1(MockEncoder):
        pass

    @encoder_registry.register(name="CustomName")
    class TestEncoder2(MockEncoder):
        pass

    @encoder_registry.register(tags=["lightweight", "fast"])
    class TestEncoder3(MockEncoder):
        pass

    return encoder_registry


# Basic registry creation and properties
def test_registry_creation():
    """Test registry creation and basic properties."""
    registry = Registry(EncoderBase, "TestRegistry")
    assert registry.name == "TestRegistry"
    assert registry.base_class == EncoderBase
    assert len(registry) == 0
    assert list(registry.list()) == []
    assert str(registry) == "TestRegistry Registry with 0 components"


# Component registration tests
def test_register_component(encoder_registry):
    """Test registering a component using the decorator."""
    @encoder_registry.register()
    class TestEncoder(MockEncoder):
        pass

    assert "TestEncoder" in encoder_registry
    assert len(encoder_registry) == 1
    assert encoder_registry.list() == ["TestEncoder"]

    # Get the registered class
    retrieved_cls = encoder_registry.get("TestEncoder")
    assert retrieved_cls == TestEncoder


def test_register_with_custom_name(encoder_registry):
    """Test registering a component with a custom name."""
    @encoder_registry.register(name="CustomEncoder")
    class TestEncoder(MockEncoder):
        pass

    assert "CustomEncoder" in encoder_registry
    assert "TestEncoder" not in encoder_registry
    assert encoder_registry.list() == ["CustomEncoder"]


def test_register_with_tags(encoder_registry):
    """Test registering a component with tags."""
    @encoder_registry.register(tags=["tag1", "tag2"])
    class TestEncoder(MockEncoder):
        pass

    assert encoder_registry.list_with_tags()["TestEncoder"] == ["tag1", "tag2"]
    assert encoder_registry.filter_by_tag("tag1") == ["TestEncoder"]
    assert encoder_registry.filter_by_tag("tag2") == ["TestEncoder"]
    assert encoder_registry.filter_by_tag("tag3") == []


def test_register_wrong_type(encoder_registry):
    """Test registering a component of the wrong type."""
    with pytest.raises(TypeError, match="must inherit from EncoderBase"):
        @encoder_registry.register()
        class WrongType:
            pass


def test_duplicate_registration(encoder_registry):
    """Test that duplicate registrations are not allowed."""
    @encoder_registry.register(name="DuplicateTest")
    class DuplicateTest(MockEncoder):
        pass

    # Intentar registrar el mismo nombre de nuevo
    with pytest.raises(ValueError, match="already registered"):
        @encoder_registry.register(name="DuplicateTest")
        class DuplicateTest2(MockEncoder):  # Test duplicate registration
            pass


# Component retrieval tests
def test_get_component(populated_registry):
    """Test getting a component by name."""
    cls = populated_registry.get("TestEncoder1")
    assert issubclass(cls, EncoderBase)

    cls = populated_registry.get("CustomName")
    assert issubclass(cls, EncoderBase)


def test_get_nonexistent_component(populated_registry):
    """Test that getting a nonexistent component raises KeyError."""
    with pytest.raises(KeyError, match="not found in registry"):
        populated_registry.get("NonexistentComponent")


def test_component_instantiation(populated_registry):
    """Test instantiating a component from the registry."""
    instance = populated_registry.instantiate("TestEncoder1", in_channels=3)
    assert isinstance(instance, EncoderBase)
    assert instance.in_channels == 3

    # With custom constructor args
    instance = populated_registry.instantiate("TestEncoder1", in_channels=10)
    assert instance.in_channels == 10


# Component listing and filtering tests
def test_list_components(populated_registry):
    """Test listing components in the registry."""
    components = populated_registry.list()
    assert len(components) == 3
    assert set(components) == {"TestEncoder1", "CustomName", "TestEncoder3"}


def test_list_with_tags(populated_registry):
    """Test listing components with their tags."""
    tags_dict = populated_registry.list_with_tags()
    assert len(tags_dict) == 3
    assert tags_dict["TestEncoder1"] == []
    assert tags_dict["CustomName"] == []
    assert set(tags_dict["TestEncoder3"]) == {"lightweight", "fast"}


def test_filter_by_tag(populated_registry):
    """Test filtering components by tag."""
    lightweight_components = populated_registry.filter_by_tag("lightweight")
    assert lightweight_components == ["TestEncoder3"]

    fast_components = populated_registry.filter_by_tag("fast")
    assert fast_components == ["TestEncoder3"]

    # Non-existent tag
    assert populated_registry.filter_by_tag("nonexistent") == []


# Special methods tests
def test_contains(populated_registry):
    """Test the __contains__ method."""
    assert "TestEncoder1" in populated_registry
    assert "CustomName" in populated_registry
    assert "NonexistentComponent" not in populated_registry


def test_len(populated_registry):
    """Test the __len__ method."""
    assert len(populated_registry) == 3


def test_repr(populated_registry):
    """Test the __repr__ method."""
    assert repr(populated_registry) == "Encoder Registry with 3 components"


# Multiple registry tests
def test_multiple_registries():
    """Test that multiple registries can coexist."""
    encoder_reg = Registry(EncoderBase, "EncoderReg")
    bottleneck_reg = Registry(BottleneckBase, "BottleneckReg")
    decoder_reg = Registry(DecoderBase, "DecoderReg")

    @encoder_reg.register()
    class TestEncoder(MockEncoder):
        pass

    @bottleneck_reg.register()
    class TestBottleneck(MockBottleneck):
        pass

    @decoder_reg.register()
    class TestDecoder(MockDecoder):
        pass

    assert "TestEncoder" in encoder_reg
    assert "TestBottleneck" in bottleneck_reg
    assert "TestDecoder" in decoder_reg

    assert "TestEncoder" not in bottleneck_reg
    assert "TestBottleneck" not in decoder_reg


# Integration tests
def test_typical_usage_flow():
    """Test a typical usage flow with registration and instantiation."""
    # Create registries for each component type
    encoder_reg = Registry(EncoderBase, "Encoder")
    bottleneck_reg = Registry(BottleneckBase, "Bottleneck")
    decoder_reg = Registry(DecoderBase, "Decoder")

    # Register components
    @encoder_reg.register(tags=["lightweight"])
    class SimpleEncoder(MockEncoder):
        pass

    @bottleneck_reg.register()
    class SimpleBottleneck(MockBottleneck):
        pass

    @decoder_reg.register()
    class SimpleDecoder(MockDecoder):
        pass

    # Instantiate components from registry
    encoder = encoder_reg.instantiate("SimpleEncoder", in_channels=3)
    bottleneck = bottleneck_reg.instantiate(
        "SimpleBottleneck", in_channels=encoder.out_channels
    )
    decoder = decoder_reg.instantiate(
        "SimpleDecoder",
        in_channels=bottleneck.out_channels,
        skip_channels=list(reversed(encoder.skip_channels))
    )

    # Verify instantiation worked correctly
    assert isinstance(encoder, EncoderBase)
    assert isinstance(bottleneck, BottleneckBase)
    assert isinstance(decoder, DecoderBase)

    # Verify components are correctly connected
    assert encoder.out_channels == bottleneck.in_channels
    assert bottleneck.out_channels == decoder.in_channels
    assert decoder.skip_channels == MockDecoder.skip_channels_expected()
