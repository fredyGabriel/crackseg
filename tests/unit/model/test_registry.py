# pyright: reportUnusedClass=false

"""Unit tests for the model component registry system."""

from typing import Any

import pytest
import torch

from crackseg.model import BottleneckBase, DecoderBase, EncoderBase
from crackseg.model.factory.registry import Registry

# Constants for test values
IN_CHANNELS_RGB = 3
MOCK_SKIP_CHANNELS = [16, 32]
OUT_CHANNELS_ENCODER = 64
OUT_CHANNELS_BOTTLENECK = 128
IN_CHANNELS_DECODER = 64
OUT_CHANNELS_DECODER = 1
IN_CHANNELS_CUSTOM = 10
NUM_ENCODERS_REGISTERED = 3
REGISTRY_REPR_EXPECTED = (
    f"Encoder Registry with {NUM_ENCODERS_REGISTERED} components"
)


class MockEncoder(EncoderBase):
    """Mock implementation of EncoderBase for testing."""

    def __init__(
        self,
        in_channels: int = IN_CHANNELS_RGB,
        skip_channels: list[int] | None = None,
    ):
        super().__init__(in_channels)
        self._skip_channels = (
            skip_channels if skip_channels is not None else MOCK_SKIP_CHANNELS
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        return x, []

    @property
    def out_channels(self) -> int:
        return OUT_CHANNELS_ENCODER

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

        # Mock encoder with standard reduction factors
        for i, channels in enumerate(self._skip_channels):
            reduction_factor = 2 ** (i + 1)
            feature_info.append(
                {
                    "channels": channels,
                    "reduction": reduction_factor,
                    "stage": i,
                }
            )

        # Add bottleneck info (final output)
        feature_info.append(
            {
                "channels": self.out_channels,
                "reduction": 2 ** len(self._skip_channels),
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
    """Mock implementation of BottleneckBase for testing."""

    def __init__(self, in_channels: int = OUT_CHANNELS_ENCODER):
        super().__init__(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    @property
    def out_channels(self) -> int:
        return OUT_CHANNELS_BOTTLENECK


class MockDecoder(DecoderBase):
    """
    MockDecoder expects skip_channels in reverse order to MockEncoder.
    If MockEncoder.skip_channels = [16, 32],
    then MockDecoder.skip_channels = [32, 16].
    """

    def __init__(
        self,
        in_channels: int = IN_CHANNELS_DECODER,
        skip_channels: list[int] | None = None,
    ):
        if skip_channels is None:
            skip_channels = list(reversed(MOCK_SKIP_CHANNELS))
        super().__init__(in_channels, skip_channels=skip_channels)
        self._out_channels: int = OUT_CHANNELS_DECODER

    def forward(
        self, x: torch.Tensor, skips: list[torch.Tensor]
    ) -> torch.Tensor:
        return x

    @property
    def out_channels(self) -> int:
        return int(self._out_channels)

    @classmethod
    def skip_channels_expected(cls) -> list[int]:
        """
        Helper for tests: Return the expected skip_channels list after
        reversal.
        """
        return list(reversed(MOCK_SKIP_CHANNELS))


# Fixtures for common test setups
@pytest.fixture
def encoder_registry() -> Registry[Any]:
    """Create a new EncoderBase registry for testing."""
    return Registry(EncoderBase, "Encoder")


@pytest.fixture
def populated_registry(encoder_registry: Registry[Any]) -> Registry[Any]:
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
    assert list(registry.list_components()) == []
    assert str(registry) == "TestRegistry Registry with 0 components"


# Component registration tests
def test_register_component(encoder_registry: Registry[Any]) -> None:
    """Test registering a component using the decorator."""

    @encoder_registry.register()
    class TestEncoder(MockEncoder):
        pass

    assert "TestEncoder" in encoder_registry
    assert len(encoder_registry) == 1
    assert encoder_registry.list_components() == ["TestEncoder"]

    # Get the registered class
    retrieved_cls = encoder_registry.get("TestEncoder")
    assert retrieved_cls == TestEncoder


def test_register_with_custom_name(encoder_registry: Registry[Any]) -> None:
    """Test registering a component with a custom name."""

    @encoder_registry.register(name="CustomEncoder")
    class TestEncoder(MockEncoder):
        pass

    assert "CustomEncoder" in encoder_registry
    assert "TestEncoder" not in encoder_registry
    assert encoder_registry.list_components() == ["CustomEncoder"]


def test_register_with_tags(encoder_registry: Registry[Any]) -> None:
    """Test registering a component with tags."""

    @encoder_registry.register(tags=["tag1", "tag2"])
    class TestEncoder(MockEncoder):
        pass

    assert encoder_registry.list_with_tags()["TestEncoder"] == ["tag1", "tag2"]
    assert encoder_registry.filter_by_tag("tag1") == ["TestEncoder"]
    assert encoder_registry.filter_by_tag("tag2") == ["TestEncoder"]
    assert encoder_registry.filter_by_tag("tag3") == []


def test_register_wrong_type(encoder_registry: Registry[Any]) -> None:
    """Test registering a component of the wrong type."""
    with pytest.raises(TypeError, match="must inherit from EncoderBase"):

        @encoder_registry.register()
        class WrongType:
            pass


def test_duplicate_registration(encoder_registry: Registry[Any]) -> None:
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
def test_get_component(populated_registry: Registry[Any]) -> None:
    """Test getting a component by name."""
    cls = populated_registry.get("TestEncoder1")
    assert issubclass(cls, EncoderBase)

    cls = populated_registry.get("CustomName")
    assert issubclass(cls, EncoderBase)


def test_get_nonexistent_component(populated_registry: Registry[Any]) -> None:
    """Test that getting a nonexistent component raises KeyError."""
    with pytest.raises(KeyError, match="not found in registry"):
        populated_registry.get("NonexistentComponent")


def test_component_instantiation(populated_registry: Registry[Any]) -> None:
    """Test instantiating a component from the registry."""
    instance = populated_registry.instantiate(
        "TestEncoder1", in_channels=IN_CHANNELS_RGB
    )
    assert isinstance(instance, EncoderBase)
    assert instance.in_channels == IN_CHANNELS_RGB

    # With custom constructor args
    instance = populated_registry.instantiate(
        "TestEncoder1", in_channels=IN_CHANNELS_CUSTOM
    )
    assert instance.in_channels == IN_CHANNELS_CUSTOM


# Component listing and filtering tests
def test_list_components(populated_registry: Registry[Any]) -> None:
    """Test listing components in the registry."""
    components = populated_registry.list_components()
    assert len(components) == NUM_ENCODERS_REGISTERED
    assert set(components) == {"TestEncoder1", "CustomName", "TestEncoder3"}


def test_list_with_tags(populated_registry: Registry[Any]) -> None:
    """Test listing components with their tags."""
    tags_dict = populated_registry.list_with_tags()
    assert len(tags_dict) == NUM_ENCODERS_REGISTERED
    assert tags_dict["TestEncoder1"] == []
    assert tags_dict["CustomName"] == []
    assert set(tags_dict["TestEncoder3"]) == {"lightweight", "fast"}


def test_filter_by_tag(populated_registry: Registry[Any]) -> None:
    """Test filtering components by tag."""
    lightweight_components = populated_registry.filter_by_tag("lightweight")
    assert lightweight_components == ["TestEncoder3"]

    fast_components = populated_registry.filter_by_tag("fast")
    assert fast_components == ["TestEncoder3"]

    # Non-existent tag
    assert populated_registry.filter_by_tag("nonexistent") == []


# Special methods tests
def test_contains(populated_registry: Registry[Any]) -> None:
    """Test the __contains__ method."""
    assert "TestEncoder1" in populated_registry
    assert "CustomName" in populated_registry
    assert "NonexistentComponent" not in populated_registry


def test_len(populated_registry: Registry[Any]) -> None:
    """Test the __len__ method."""
    assert len(populated_registry) == NUM_ENCODERS_REGISTERED


def test_repr(populated_registry: Registry[Any]) -> None:
    """Test the __repr__ method."""
    assert repr(populated_registry) == REGISTRY_REPR_EXPECTED


# Multiple registry tests
def test_multiple_registries() -> None:
    """Test that multiple registries can coexist."""
    encoder_reg: Registry[Any] = Registry(EncoderBase, "EncoderReg")
    bottleneck_reg: Registry[Any] = Registry(BottleneckBase, "BottleneckReg")
    decoder_reg: Registry[Any] = Registry(DecoderBase, "DecoderReg")

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
    encoder = encoder_reg.instantiate(
        "SimpleEncoder", in_channels=IN_CHANNELS_RGB
    )
    bottleneck = bottleneck_reg.instantiate(
        "SimpleBottleneck", in_channels=encoder.out_channels
    )
    decoder = decoder_reg.instantiate(
        "SimpleDecoder",
        in_channels=bottleneck.out_channels,
        skip_channels=list(reversed(encoder.skip_channels)),
    )

    # Verify instantiation worked correctly
    assert isinstance(encoder, EncoderBase)
    assert isinstance(bottleneck, BottleneckBase)
    assert isinstance(decoder, DecoderBase)

    # Verify components are correctly connected
    assert encoder.out_channels == bottleneck.in_channels
    assert bottleneck.out_channels == decoder.in_channels
    assert decoder.skip_channels == MockDecoder.skip_channels_expected()
    assert encoder.in_channels == IN_CHANNELS_RGB
