"""Unit tests for model component factory functions."""

import pytest
import torch
from unittest.mock import patch, MagicMock

from src.model.base import EncoderBase, BottleneckBase, DecoderBase, UNetBase
from src.model.factory import (
    ConfigurationError, validate_config
)


# --- Test Cases for validate_config ---

def test_validate_config_empty():
    """Test validation of an empty config."""
    with pytest.raises(ConfigurationError,
                       match="Missing required configuration"):
        validate_config({}, required_keys=["type"], component_type="test")


def test_validate_config_missing_key():
    """Test validation of a config with missing required keys."""
    config = {"type": "mock"}
    with pytest.raises(
        ConfigurationError,
        match="Missing required configuration"
    ):
        validate_config(
            config,
            required_keys=["in_channels", "type"],
            component_type="test"
        )


def test_validate_config_valid():
    """Test validation of a valid config."""
    config = {
        "type": "mock",
        "in_channels": 3,
        "out_channels": 64
    }
    # Should not raise an exception
    validate_config(
        config,
        required_keys=["in_channels", "type"],
        component_type="test"
    )


# --- Test Cases for factory functions using mocks ---

@patch('src.model.factory.encoder_registry')
def test_create_encoder_basic(mock_registry):
    """Test creation of a basic encoder with mocked registry."""
    from src.model.factory import create_encoder

    # Setup mock
    mock_encoder = MagicMock(spec=EncoderBase)
    mock_registry.instantiate.return_value = mock_encoder

    # Test
    config = {
        "type": "MockEncoder",
        "in_channels": 3,
        "depth": 4
    }
    encoder = create_encoder(config)

    # Verify
    mock_registry.instantiate.assert_called_once_with(
        "MockEncoder", in_channels=3, depth=4
    )
    assert encoder == mock_encoder


@patch('src.model.factory.encoder_registry')
def test_create_encoder_invalid_type(mock_registry):
    """Test creation with an invalid encoder type."""
    from src.model.factory import create_encoder

    # Setup mock to raise KeyError
    mock_registry.instantiate.side_effect = KeyError("Not found")
    mock_registry.list.return_value = []

    config = {
        "type": "NonexistentEncoder",
        "in_channels": 3
    }

    with pytest.raises(ConfigurationError, match="not found in registry"):
        create_encoder(config)


@patch('src.model.factory.bottleneck_registry')
def test_create_bottleneck_basic(mock_registry):
    """Test creation of a basic bottleneck with mocked registry."""
    from src.model.factory import create_bottleneck

    # Setup mock
    mock_bottleneck = MagicMock(spec=BottleneckBase)
    mock_registry.instantiate.return_value = mock_bottleneck

    # Test
    config = {
        "type": "MockBottleneck",
        "in_channels": 64,
        "out_channels": 128
    }
    bottleneck = create_bottleneck(config)

    # Verify
    mock_registry.instantiate.assert_called_once_with(
        "MockBottleneck", in_channels=64, out_channels=128
    )
    assert bottleneck == mock_bottleneck


@patch('src.model.factory.decoder_registry')
def test_create_decoder_basic(mock_registry):
    """Test creation of a basic decoder with mocked registry."""
    from src.model.factory import create_decoder

    # Setup mock
    mock_decoder = MagicMock(spec=DecoderBase)
    mock_registry.instantiate.return_value = mock_decoder

    # Test
    config = {
        "type": "MockDecoder",
        "in_channels": 128,
        "out_channels": 1,
        "skip_channels": [64, 32, 16]
    }
    decoder = create_decoder(config)

    # Verify
    mock_registry.instantiate.assert_called_once_with(
        "MockDecoder", in_channels=128, out_channels=1,
        skip_channels=[64, 32, 16]
    )
    assert decoder == mock_decoder


@patch('hydra.utils.instantiate')
def test_create_unet_basic(mock_instantiate):
    """Test creation of a basic UNet model with mocked Hydra."""
    from src.model.factory import create_unet

    # Setup mock
    mock_unet = MagicMock(spec=UNetBase)
    mock_instantiate.return_value = mock_unet

    # Test
    config = {
        "_target_": "src.model.unet.BaseUNet",
        "encoder": {"mock": "encoder"},
        "bottleneck": {"mock": "bottleneck"},
        "decoder": {"mock": "decoder"}
    }

    # Patch the component creation functions to avoid validation
    with patch('src.model.factory.create_encoder') as mock_create_encoder, \
         patch('src.model.factory.create_bottleneck') as \
            mock_create_bottleneck, \
         patch('src.model.factory.create_decoder') as mock_create_decoder, \
         patch('hydra.utils.get_class') as mock_get_class:

        # Setup component mocks
        mock_encoder = MagicMock(spec=EncoderBase)
        mock_bottleneck = MagicMock(spec=BottleneckBase)
        mock_decoder = MagicMock(spec=DecoderBase)

        # Configure mock properties for validation
        mock_encoder.out_channels = 64
        mock_encoder.skip_channels = [32, 16]
        mock_bottleneck.in_channels = 64
        mock_bottleneck.out_channels = 128
        mock_decoder.in_channels = 128
        mock_decoder.skip_channels = [16, 32]  # Reversed to match encoder

        mock_create_encoder.return_value = mock_encoder
        mock_create_bottleneck.return_value = mock_bottleneck
        mock_create_decoder.return_value = mock_decoder

        # Mock the UNet class
        mock_get_class.return_value = MagicMock(return_value=mock_unet)

        # Call the function
        unet = create_unet(config)

    # Verify
    assert mock_create_encoder.called
    assert mock_create_bottleneck.called
    assert mock_create_decoder.called
    assert unet is not None


@patch('hydra.utils.instantiate')
def test_create_unet_with_final_activation(mock_instantiate):
    """Test UNet creation with a final activation function."""
    from src.model.factory import create_unet

    # Setup mock
    mock_unet = MagicMock(spec=UNetBase)
    mock_unet.final_activation = torch.nn.Sigmoid()
    mock_instantiate.return_value = mock_unet

    # Test
    config = {
        "_target_": "src.model.unet.BaseUNet",
        "encoder": {"mock": "encoder"},
        "bottleneck": {"mock": "bottleneck"},
        "decoder": {"mock": "decoder"},
        "final_activation": {
            "_target_": "torch.nn.Sigmoid"
        }
    }

    # Patch the component creation functions to avoid validation
    with patch('src.model.factory.create_encoder') as mock_create_encoder, \
         patch('src.model.factory.create_bottleneck') as \
            mock_create_bottleneck, \
         patch('src.model.factory.create_decoder') as mock_create_decoder, \
         patch('hydra.utils.get_class') as mock_get_class:

        # Setup component mocks
        mock_encoder = MagicMock(spec=EncoderBase)
        mock_bottleneck = MagicMock(spec=BottleneckBase)
        mock_decoder = MagicMock(spec=DecoderBase)

        # Configure mock properties for validation
        mock_encoder.out_channels = 64
        mock_encoder.skip_channels = [32, 16]
        mock_bottleneck.in_channels = 64
        mock_bottleneck.out_channels = 128
        mock_decoder.in_channels = 128
        mock_decoder.skip_channels = [16, 32]  # Reversed to match encoder

        mock_create_encoder.return_value = mock_encoder
        mock_create_bottleneck.return_value = mock_bottleneck
        mock_create_decoder.return_value = mock_decoder

        # Mock the UNet class
        mock_get_class.return_value = MagicMock(return_value=mock_unet)

        # Call the function
        unet = create_unet(config)

    # Verify
    assert mock_create_encoder.called
    assert mock_create_bottleneck.called
    assert mock_create_decoder.called
    assert unet == mock_unet
    assert isinstance(unet.final_activation, torch.nn.Sigmoid)
