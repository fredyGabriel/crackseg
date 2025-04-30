"""Integration tests for model instantiation and forward pass from config."""

import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from src.model.factory import create_unet
from src.model.base import UNetBase
from tests.model.test_registry import (
    MockEncoder, MockBottleneck, MockDecoder
)

# Ensure mock components are registered before tests run
# pylint: disable=unused-import
import tests.model.test_factory  # noqa: F401

# --- Helper Function to Load Config ---


def load_test_config(config_name: str = "model/unet_mock") -> DictConfig:
    """Loads a specific test configuration using Hydra Compose API."""
    # Determine the absolute path to the configs directory
    # Assumes tests are run from the project root or tests/ are siblings of
    # configs/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.abspath(os.path.join(script_dir, "..", "..",
                                               "configs"))

    if not os.path.exists(config_path):
        # Fallback if run from a different working directory
        # (e.g., project root)
        config_path = os.path.abspath(os.path.join(os.getcwd(), "configs"))
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config directory not found relative to \
test or cwd: {config_path}")

    # Use Hydra's Compose API for loading
    hydra.initialize_config_dir(config_dir=config_path, version_base=None)
    cfg = hydra.compose(config_name=config_name)
    # Clean up hydra state
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    return cfg

# --- Test Cases ---


def test_unet_instantiation_from_manual_config():
    """Test instantiating the UNet model by manually loading config."""
    cfg = load_test_config()
    config_dict = OmegaConf.to_container(cfg.model, resolve=True)
    unet = create_unet(config_dict)

    assert isinstance(unet, UNetBase)
    assert isinstance(unet.encoder, MockEncoder)
    assert isinstance(unet.bottleneck, MockBottleneck)
    assert isinstance(unet.decoder, MockDecoder)
    assert unet.encoder.in_channels == cfg.model.encoder.in_channels
    assert unet.bottleneck.in_channels == cfg.model.bottleneck.in_channels
    assert unet.decoder.in_channels == cfg.model.decoder.in_channels
    assert unet.decoder.skip_channels == cfg.model.decoder.skip_channels

    if cfg.model.get("final_activation"):
        assert unet.final_activation is not None
        activation_cls = hydra.utils.get_class(
            cfg.model.final_activation._target_)
        assert isinstance(unet.final_activation, activation_cls)
    else:
        assert unet.final_activation is None


def test_unet_forward_pass_from_manual_config():
    """Test the forward pass of a UNet instantiated from manually loaded
    config."""
    cfg = load_test_config()
    config_dict = OmegaConf.to_container(cfg.model, resolve=True)
    unet = create_unet(config_dict)
    unet.eval()

    input_channels = unet.get_input_channels()
    x = torch.randn(2, input_channels, 64, 64)

    with torch.no_grad():
        output = unet(x)

    assert output.shape[0] == x.shape[0]
    assert output.shape[1] == input_channels  # Correct for MockDecoder
    assert output.shape[2:] == x.shape[2:]

    if unet.final_activation:
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)
