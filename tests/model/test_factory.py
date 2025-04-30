"""Unit tests for model component factory functions."""

import pytest

from src.model.base import EncoderBase, BottleneckBase, DecoderBase, UNetBase
from src.model.factory import (
    ConfigurationError, validate_config,
    create_encoder, create_bottleneck, create_decoder, create_unet,
    create_component_from_config,
    encoder_registry, bottleneck_registry, decoder_registry
)

# Reusing mock classes from registry tests
from tests.model.test_registry import (
    MockEncoder, MockBottleneck, MockDecoder
)


# Mock UNet class for testing - REMOVED as BaseUNet is used directly now
# class MockUNet(UNetBase): ...


# Register the mock components in their respective registries
@encoder_registry.register()
class TestEncoder(MockEncoder):
    pass


@bottleneck_registry.register()
class TestBottleneck(MockBottleneck):
    pass


@decoder_registry.register()
class TestDecoder(MockDecoder):
    pass


# @unet_registry.register() # REMOVED registration
# class TestUNet(MockUNet):
#     pass


# Sample configurations for testing
@pytest.fixture
def encoder_config():
    return {
        "type": "TestEncoder",
        "in_channels": 3
    }


@pytest.fixture
def bottleneck_config():
    return {
        "type": "TestBottleneck",
        "in_channels": 64
    }


@pytest.fixture
def decoder_config():
    return {
        "type": "TestDecoder",
        "in_channels": 128,
        "skip_channels": [16, 32]
    }


@pytest.fixture
def unet_config(encoder_config, bottleneck_config, decoder_config):
    return {
        # "type": "TestUNet",  # No longer needed if using _target_
        "_target_": "src.model.unet.BaseUNet",  # Target the actual implem.
        "encoder": encoder_config,
        "bottleneck": bottleneck_config,
        "decoder": decoder_config,
        # Example extra param for BaseUNet if it accepted it
        "dropout_rate": 0.2,
        "final_activation": {"_target_": "torch.nn.Sigmoid"}
    }


# Tests for validate_config
def test_validate_config_with_valid_config():
    """Test that validate_config passes with all required keys."""
    config = {"key1": "value1", "key2": "value2"}
    # Should not raise an exception
    validate_config(config, ["key1", "key2"], "test")


def test_validate_config_with_missing_keys():
    """Test that validate_config raises ConfigurationError for missing keys."""
    config = {"key1": "value1"}
    with pytest.raises(ConfigurationError, match="Missing required \
configuration"):
        validate_config(config, ["key1", "key2"], "test")


# Tests for create_encoder
def test_create_encoder_with_valid_config(encoder_config):
    """Test creating an encoder with valid configuration."""
    encoder = create_encoder(encoder_config)
    assert isinstance(encoder, EncoderBase)
    assert encoder.in_channels == 3


def test_create_encoder_with_missing_keys():
    """Test that create_encoder raises ConfigurationError for missing keys."""
    # Missing in_channels
    config = {"type": "TestEncoder"}
    with pytest.raises(ConfigurationError, match="Missing required \
configuration"):
        create_encoder(config)


def test_create_encoder_with_invalid_type():
    """Test that create_encoder raises ConfigurationError for invalid type."""
    config = {"type": "NonExistentEncoder", "in_channels": 3}
    with pytest.raises(ConfigurationError, match="not found in registry"):
        create_encoder(config)


# Tests for create_bottleneck
def test_create_bottleneck_with_valid_config(bottleneck_config):
    """Test creating a bottleneck with valid configuration."""
    bottleneck = create_bottleneck(bottleneck_config)
    assert isinstance(bottleneck, BottleneckBase)
    assert bottleneck.in_channels == 64


def test_create_bottleneck_with_missing_keys():
    """Test that create_bottleneck raises ConfigurationError for missing keys.
    """
    # Missing in_channels
    config = {"type": "TestBottleneck"}
    with pytest.raises(ConfigurationError, match="Missing required \
configuration"):
        create_bottleneck(config)


def test_create_bottleneck_with_invalid_type():
    """Test that create_bottleneck raises ConfigurationError for invalid type.
    """
    config = {"type": "NonExistentBottleneck", "in_channels": 64}
    with pytest.raises(ConfigurationError, match="not found in registry"):
        create_bottleneck(config)


# Tests for create_decoder
def test_create_decoder_with_valid_config(decoder_config):
    """Test creating a decoder with valid configuration."""
    decoder = create_decoder(decoder_config)
    assert isinstance(decoder, DecoderBase)
    assert decoder.in_channels == 128
    assert decoder.skip_channels == [16, 32]


def test_create_decoder_with_missing_keys():
    """Test that create_decoder raises ConfigurationError for missing keys."""
    # Missing skip_channels
    config = {"type": "TestDecoder", "in_channels": 128}
    with pytest.raises(ConfigurationError, match="Missing required \
configuration"):
        create_decoder(config)


def test_create_decoder_with_invalid_type():
    """Test that create_decoder raises ConfigurationError for invalid type."""
    config = {
        "type": "NonExistentDecoder",
        "in_channels": 128,
        "skip_channels": [16, 32]
    }
    with pytest.raises(ConfigurationError, match="not found in registry"):
        create_decoder(config)


# Tests for create_unet
def test_create_unet_with_valid_config(unet_config):
    """Test creating a UNet with valid configuration."""
    unet = create_unet(unet_config)
    assert isinstance(unet, UNetBase)
    # assert unet.dropout_rate == 0.2  # BaseUNet doesn't have dropout_rate

    # Verify component connections
    assert isinstance(unet.encoder, EncoderBase)
    assert isinstance(unet.bottleneck, BottleneckBase)
    assert isinstance(unet.decoder, DecoderBase)

    # Verify channel compatibility
    assert unet.encoder.out_channels == unet.bottleneck.in_channels
    assert unet.bottleneck.out_channels == unet.decoder.in_channels
    assert unet.encoder.skip_channels == unet.decoder.skip_channels


def test_create_unet_with_default_type(encoder_config, bottleneck_config,
                                       decoder_config):
    """Test creating a UNet when _target_ is missing (should fail)."""
    # Create config without _target_ key
    config = {
        "encoder": encoder_config,
        "bottleneck": bottleneck_config,
        "decoder": decoder_config
    }

    # Should fail because _target_ is now required
    with pytest.raises(ConfigurationError,
                       match="Missing required configuration"):
        create_unet(config)


def test_create_unet_with_missing_components():
    """Test that create_unet raises ConfigurationError for missing components.
    """
    # Missing decoder
    config = {
        "_target_": "src.model.unet.BaseUNet",
        "encoder": {"type": "TestEncoder", "in_channels": 3},
        "bottleneck": {"type": "TestBottleneck", "in_channels": 64}
    }
    with pytest.raises(ConfigurationError,
                       match="Missing required configuration"):
        create_unet(config)


def test_create_unet_with_invalid_component_config(unet_config):
    """Test that create_unet propagates ConfigurationError from components."""
    # Invalid encoder configuration (missing in_channels)
    invalid_config = unet_config.copy()
    invalid_config["encoder"] = {"type": "TestEncoder",
                                 "_target_":
                                 "tests.model.test_registry.MockEncoder"}

    with pytest.raises(ConfigurationError,
                       match="Missing required configuration"):
        create_unet(invalid_config)


def test_create_unet_with_invalid_target():
    """Test creating UNet with an invalid _target_ class path."""
    config = {
        "_target_": "invalid.path.NonExistentUNet",
        "encoder": {"type": "TestEncoder", "in_channels": 3,
                    "_target_": "tests.model.test_registry.MockEncoder"},
        "bottleneck": {"type": "TestBottleneck", "in_channels": 64,
                       "_target_": "tests.model.test_registry.MockBottleneck"},
        "decoder": {"type": "TestDecoder", "in_channels": 128,
                    "skip_channels": [16, 32],
                    "_target_": "tests.model.test_registry.MockDecoder"}
    }
    with pytest.raises(ConfigurationError, match="Could not import UNet class"
                       ):
        create_unet(config)


def test_create_unet_with_non_unet_target():
    """Test creating UNet with a _target_ that is not a UNetBase subclass."""
    config = {
        "_target_": "torch.nn.Module",  # Invalid target class
        "encoder": {"type": "TestEncoder", "in_channels": 3,
                    "_target_": "tests.model.test_registry.MockEncoder"},
        "bottleneck": {"type": "TestBottleneck", "in_channels": 64,
                       "_target_": "tests.model.test_registry.MockBottleneck"},
        "decoder": {"type": "TestDecoder", "in_channels": 128,
                    "skip_channels": [16, 32],
                    "_target_": "tests.model.test_registry.MockDecoder"}
    }
    with pytest.raises(ConfigurationError,
                       match="does not inherit from UNetBase"):
        create_unet(config)


# Tests for create_component_from_config
def test_create_component_from_config():
    """Test the generic component creation function."""
    config = {"type": "TestEncoder", "in_channels": 3}
    component = create_component_from_config(config, encoder_registry)
    assert isinstance(component, EncoderBase)
    assert component.in_channels == 3


def test_create_component_from_config_with_missing_type():
    """Test that create_component_from_config raises error for missing type."""
    config = {"in_channels": 3}
    with pytest.raises(ConfigurationError, match="Missing required \
configuration"):
        create_component_from_config(config, encoder_registry)


def test_create_component_from_config_with_invalid_type():
    """Test that create_component_from_config raises error for invalid type."""
    config = {"type": "NonExistentComponent", "in_channels": 3}
    with pytest.raises(ConfigurationError, match="not found in registry"):
        create_component_from_config(config, encoder_registry)
