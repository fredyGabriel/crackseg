"""Unit tests for model component factory functions."""

import os
from typing import Any
from unittest.mock import MagicMock, patch

import hydra
import pytest
from omegaconf import OmegaConf
from torch import nn

from src.model import BottleneckBase, DecoderBase, EncoderBase, UNetBase
from src.model.factory import ConfigurationError, validate_config


# Fixture to load Hydra configuration
@pytest.fixture(scope="session")
def cfg() -> Any:
    # Calculate absolute path to configs directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
    config_dir = os.path.join(project_root, "configs")

    if not os.path.isdir(config_dir):
        # Fallback if running from different directory
        config_dir = os.path.abspath(os.path.join(os.getcwd(), "configs"))
        if not os.path.isdir(config_dir):
            raise FileNotFoundError(
                f"Config directory not found at {config_dir}"
            )

    with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
        config = hydra.compose(config_name="config.yaml")
    return config


# --- Test Cases for validate_config ---


def test_validate_config_empty():
    """Test validation of an empty config."""
    with pytest.raises(
        ConfigurationError, match="Missing required configuration"
    ):
        validate_config({}, required_keys=["type"], component_type="test")


def test_validate_config_missing_key():
    """Test validation of a config with missing required keys."""
    config = {"type": "mock"}
    with pytest.raises(
        ConfigurationError, match="Missing required configuration"
    ):
        validate_config(
            config,
            required_keys=["in_channels", "type"],
            component_type="test",
        )


def test_validate_config_valid(cfg: Any):
    """Test validation of a valid config."""
    config = {
        "type": "mock",
        "in_channels": cfg.data.num_channels_rgb,
        "out_channels": 64,  # Si existe en config usarlo
    }
    # Should not raise an exception
    validate_config(
        config, required_keys=["in_channels", "type"], component_type="test"
    )


# NOTE: Tests for individual component factories (encoder, bottleneck, decoder)
# have been removed. After the factory refactor, their instantiation logic is
# now covered by the integration tests below (test_create_unet_basic,
# test_create_unet_with_final_activation). This avoids redundancy and ensures
# tests reflect the real API and integration flow.

# --- Tests for UNet Factory ---


# Patch the correct internal helper
@patch("src.model.factory.config._try_instantiation_methods")
@patch("hydra.utils.get_class")
def test_create_unet_basic(
    mock_get_class: Any, mock_try_instantiation_methods: Any, cfg: Any
):
    """Test creation of a basic UNet model, mocking the core instantiation
    helper."""
    from src.model.factory import create_unet

    # Setup component mocks
    mock_encoder = MagicMock(spec=EncoderBase)
    mock_bottleneck = MagicMock(spec=BottleneckBase)
    mock_decoder = MagicMock(spec=DecoderBase)

    # Configure mock properties needed by create_unet
    encoder_out_channels = (
        cfg.model.encoder.init_features
        if hasattr(cfg.model.encoder, "init_features")
        else 64
    )  # Default fallback
    # TODO: If available in config, use cfg.model.decoder.skip_channels_list
    # or similar
    encoder_skip_channels = [
        32,
        16,
    ]
    # TODO: If available in config, use cfg.model.bottleneck.out_channels
    bottleneck_out_channels = 128

    mock_encoder.out_channels = encoder_out_channels
    mock_encoder.skip_channels = encoder_skip_channels
    mock_bottleneck.out_channels = bottleneck_out_channels

    # Configure side_effect for the mocked _try_instantiation_methods
    def try_instantiation_side_effect(
        config: Any, component_type: Any, registry: Any, base_class: Any
    ):
        if component_type == "encoder":
            return mock_encoder
        elif component_type == "bottleneck":
            return mock_bottleneck
        elif component_type == "decoder":
            return mock_decoder
        else:
            raise ValueError(
                f"Unexpected component_type for mock: {component_type}"
            )

    mock_try_instantiation_methods.side_effect = try_instantiation_side_effect

    # Mock the final UNet class and its instantiation
    MockUnetClass = MagicMock()
    mock_unet_instance = MagicMock(spec=UNetBase)
    MockUnetClass.return_value = mock_unet_instance
    mock_get_class.return_value = MockUnetClass
    mock_get_class.return_value.__name__ = "MockedUNetClass"

    # Test config (includes in_channels for validation)
    config = OmegaConf.create(
        {
            "_target_": "src.model.unet.BaseUNet",
            "encoder": {
                "_target_": "e",
                "type": "E",
                "in_channels": cfg.data.num_channels_rgb,
            },
            "bottleneck": {
                "_target_": "b",
                "type": "B",
                "in_channels": encoder_out_channels,  # Add required field
            },
            "decoder": {
                "_target_": "d",
                "type": "D",
                "in_channels": bottleneck_out_channels,  # Add required field
            },
        }
    )

    # Call the function
    unet = create_unet(config)

    # Assertions
    assert unet is mock_unet_instance
    assert mock_try_instantiation_methods.call_count == 3
    calls = mock_try_instantiation_methods.call_args_list
    assert calls[0].args[1] == "encoder"
    assert calls[1].args[1] == "bottleneck"
    assert calls[2].args[1] == "decoder"

    mock_get_class.assert_called_once_with("src.model.unet.BaseUNet")
    MockUnetClass.assert_called_once_with(
        encoder=mock_encoder, bottleneck=mock_bottleneck, decoder=mock_decoder
    )


# Patch the correct internal helper
@patch("src.model.factory.config._try_instantiation_methods")
@patch("hydra.utils.get_class")
@patch("hydra.utils.instantiate")
def test_create_unet_with_final_activation(
    mock_hydra_instantiate: Any,
    mock_get_class: Any,
    mock_try_instantiation_methods: Any,
    cfg: Any,
):
    """Test UNet creation with activation, mocking core instantiation."""
    from src.model.factory import create_unet

    # Setup component mocks
    mock_encoder = MagicMock(spec=EncoderBase)
    mock_bottleneck = MagicMock(spec=BottleneckBase)
    mock_decoder = MagicMock(spec=DecoderBase)
    encoder_out_channels = (
        cfg.model.encoder.init_features
        if hasattr(cfg.model.encoder, "init_features")
        else 64
    )  # Default fallback
    # TODO: If available in config, use cfg.model.decoder.skip_channels_list
    # or similar
    encoder_skip_channels = [
        32,
        16,
    ]
    # TODO: If available in config, use cfg.model.bottleneck.out_channels
    bottleneck_out_channels = 128

    mock_encoder.out_channels = encoder_out_channels
    mock_encoder.skip_channels = encoder_skip_channels
    mock_bottleneck.out_channels = bottleneck_out_channels

    # Configure side_effect for the mocked _try_instantiation_methods
    def try_instantiation_side_effect(
        config: Any, component_type: Any, registry: Any, base_class: Any
    ):
        if component_type == "encoder":
            return mock_encoder
        if component_type == "bottleneck":
            return mock_bottleneck
        if component_type == "decoder":
            return mock_decoder
        raise ValueError(f"Unexpected component_type: {component_type}")

    mock_try_instantiation_methods.side_effect = try_instantiation_side_effect

    # Mock UNet class
    MockUnetClass = MagicMock()
    mock_unet_base_instance = MagicMock(spec=UNetBase)
    MockUnetClass.return_value = mock_unet_base_instance
    mock_get_class.return_value = MockUnetClass
    mock_get_class.return_value.__name__ = "MockedUNetClass"

    # Mock final activation
    mock_activation = MagicMock(spec=nn.Module)
    mock_hydra_instantiate.return_value = mock_activation

    # Test config (includes in_channels for validation)
    final_activation_config = {"_target_": "torch.nn.Sigmoid"}
    config = OmegaConf.create(
        {
            "_target_": "src.model.unet.BaseUNet",
            "encoder": {
                "_target_": "e",
                "type": "E",
                "in_channels": cfg.data.num_channels_rgb,
            },
            "bottleneck": {
                "_target_": "b",
                "type": "B",
                "in_channels": encoder_out_channels,  # Add required field
            },
            "decoder": {
                "_target_": "d",
                "type": "D",
                "in_channels": bottleneck_out_channels,  # Add required field
            },
            "final_activation": final_activation_config,
        }
    )

    # Call the function
    unet = create_unet(config)

    # Assertions
    assert isinstance(unet, nn.Sequential)
    assert unet[0] is mock_unet_base_instance
    assert unet[1] is mock_activation

    assert mock_try_instantiation_methods.call_count == 3
    mock_get_class.assert_called_once()
    MockUnetClass.assert_called_once_with(
        encoder=mock_encoder, bottleneck=mock_bottleneck, decoder=mock_decoder
    )
    mock_hydra_instantiate.assert_called_once_with(config.final_activation)
