"""Unit tests for model component factory functions."""

import pytest
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf
import torch.nn as nn
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


# NOTE: Tests for individual component factories (encoder, bottleneck, decoder)
# have been removed. After the factory refactor, their instantiation logic is
# now covered by the integration tests below (test_create_unet_basic,
# test_create_unet_with_final_activation). This avoids redundancy and ensures
# tests reflect the real API and integration flow.

# --- Tests for UNet Factory ---

# Patch the internal helper
@patch('src.model.config.instantiation._instantiate_component')
@patch('hydra.utils.get_class')
def test_create_unet_basic(
    mock_get_class, mock_instantiate_component
):
    """Test creation of a basic UNet model, mocking the core instantiation
    helper."""
    from src.model.factory import create_unet

    # Setup component mocks
    mock_encoder = MagicMock(spec=EncoderBase)
    mock_bottleneck = MagicMock(spec=BottleneckBase)
    mock_decoder = MagicMock(spec=DecoderBase)

    # Configure mock properties needed by create_unet
    mock_encoder.out_channels = 64
    mock_encoder.skip_channels = [32, 16]
    mock_bottleneck.out_channels = 128  # Needed for decoder runtime_params

    # Configure side_effect for the mocked internal helper
    def instantiate_side_effect(*args, **kwargs):
        category = kwargs.get('component_category') or (
            args[2] if len(args) > 2 else None)
        if category == 'encoder':
            return mock_encoder
        elif category == 'bottleneck':
            return mock_bottleneck
        elif category == 'decoder':
            return mock_decoder
        else:
            raise ValueError(f"Unexpected category for mock: {category}")
    mock_instantiate_component.side_effect = instantiate_side_effect

    # Mock the final UNet class and its instantiation
    MockUnetClass = MagicMock()
    mock_unet_instance = MagicMock(spec=UNetBase)
    MockUnetClass.return_value = mock_unet_instance
    mock_get_class.return_value = MockUnetClass
    mock_get_class.return_value.__name__ = "MockedUNetClass"

    # Test config (content less critical now, as instantiation is mocked)
    config = OmegaConf.create({
        "_target_": "src.model.unet.BaseUNet",
        "encoder": {"_target_": "e", "type": "E"},
        "bottleneck": {"_target_": "b", "type": "B"},
        "decoder": {"_target_": "d", "type": "D"}
    })

    # Call the function
    unet = create_unet(config)

    # Assertions
    assert unet is mock_unet_instance
    # Check calls based on category passed to the mocked helper
    assert mock_instantiate_component.call_count == 3
    calls = mock_instantiate_component.call_args_list
    # Check encoder call
    assert calls[0].kwargs['component_category'] == 'encoder'
    assert calls[0].kwargs['config'] == {"_target_": "e", "type": "E"}
    # Check bottleneck call
    assert calls[1].kwargs['component_category'] == 'bottleneck'
    assert calls[1].kwargs['config'] == {"_target_": "b", "type": "B"}
    assert calls[1].kwargs['runtime_params'] == {"in_channels": 64}
    # Check decoder call
    assert calls[2].kwargs['component_category'] == 'decoder'
    assert calls[2].kwargs['config'] == {"_target_": "d", "type": "D"}
    assert calls[2].kwargs['runtime_params'] == {
        "in_channels": 128,
        "skip_channels_list": [16, 32]
    }

    mock_get_class.assert_called_once_with("src.model.unet.BaseUNet")
    MockUnetClass.assert_called_once_with(
        encoder=mock_encoder, bottleneck=mock_bottleneck, decoder=mock_decoder
    )


# Patch internal helper
@patch('src.model.config.instantiation._instantiate_component')
@patch('hydra.utils.get_class')
@patch('hydra.utils.instantiate')
def test_create_unet_with_final_activation(
    mock_hydra_instantiate, mock_get_class, mock_instantiate_component
):
    """Test UNet creation with activation, mocking core instantiation."""
    from src.model.factory import create_unet

    # Setup component mocks
    mock_encoder = MagicMock(spec=EncoderBase)
    mock_bottleneck = MagicMock(spec=BottleneckBase)
    mock_decoder = MagicMock(spec=DecoderBase)
    mock_encoder.out_channels = 64
    mock_encoder.skip_channels = [32, 16]
    mock_bottleneck.out_channels = 128

    # Configure side_effect for the mocked internal helper
    def instantiate_side_effect(*args, **kwargs):
        category = kwargs.get('component_category') or (
            args[2] if len(args) > 2 else None)
        if category == 'encoder':
            return mock_encoder
        if category == 'bottleneck':
            return mock_bottleneck
        if category == 'decoder':
            return mock_decoder
        raise ValueError(f"Unexpected category: {category}")
    mock_instantiate_component.side_effect = instantiate_side_effect

    # Mock UNet class
    MockUnetClass = MagicMock()
    mock_unet_base_instance = MagicMock(spec=UNetBase)
    MockUnetClass.return_value = mock_unet_base_instance
    mock_get_class.return_value = MockUnetClass
    mock_get_class.return_value.__name__ = "MockedUNetClass"

    # Mock final activation
    mock_activation = MagicMock(spec=nn.Module)
    mock_hydra_instantiate.return_value = mock_activation

    # Test config
    final_activation_config = {"_target_": "torch.nn.Sigmoid"}
    config = OmegaConf.create({
        "_target_": "src.model.unet.BaseUNet",
        "encoder": {"_target_": "e", "type": "E"},
        "bottleneck": {"_target_": "b", "type": "B"},
        "decoder": {"_target_": "d", "type": "D"},
        "final_activation": final_activation_config
    })

    # Call the function
    unet = create_unet(config)

    # Assertions
    assert isinstance(unet, nn.Sequential)
    assert unet[0] is mock_unet_base_instance
    assert unet[1] is mock_activation

    assert mock_instantiate_component.call_count == 3
    mock_get_class.assert_called_once()
    MockUnetClass.assert_called_once_with(
        encoder=mock_encoder, bottleneck=mock_bottleneck, decoder=mock_decoder
    )
    mock_hydra_instantiate.assert_called_once_with(config.final_activation)
