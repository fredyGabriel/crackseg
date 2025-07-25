from typing import cast

import pytest
import torch.nn as nn
from omegaconf import OmegaConf

from crackseg.model.components.cbam import CBAM
from crackseg.model.factory.factory import insert_cbam_if_enabled


def test_cbam_enabled_in_config():
    """Test CBAM is inserted when enabled in config."""
    dummy = nn.Identity()
    config = OmegaConf.create(
        {
            "cbam_enabled": True,
            "cbam_params": {
                "reduction": 1,
                "kernel_size": 3,
                "in_channels": 4,
            },
        }
    )
    out = insert_cbam_if_enabled(dummy, config)
    assert isinstance(out, nn.Sequential)
    assert any(isinstance(m, CBAM) for m in out.modules())


def test_cbam_disabled_in_config():
    """Test CBAM is not inserted when disabled in config."""
    dummy = nn.Identity()
    config = OmegaConf.create(
        {
            "cbam_enabled": False,
            "cbam_params": {
                "reduction": 1,
                "kernel_size": 3,
                "in_channels": 4,
            },
        }
    )
    out = insert_cbam_if_enabled(dummy, config)
    assert out is dummy


def test_cbam_custom_params():
    """Test CBAM is created with custom parameters from config."""
    dummy = nn.Identity()
    config = OmegaConf.create(
        {
            "cbam_enabled": True,
            "cbam_params": {
                "reduction": 2,
                "kernel_size": 5,
                "in_channels": 4,
            },
        }
    )
    out = insert_cbam_if_enabled(dummy, config)
    cbam = [m for m in out.modules() if isinstance(m, CBAM)][0]
    cbam_module = cast(CBAM, cbam)

    # Cast the attention modules to help type checker
    channel_attn = cast(nn.Module, cbam_module.channel_attn)
    spatial_attn = cast(nn.Module, cbam_module.spatial_attn)

    assert (
        hasattr(channel_attn, "reduction") and channel_attn.reduction == 2
    )  # noqa: PLR2004
    assert hasattr(spatial_attn, "conv") and hasattr(
        spatial_attn.conv, "kernel_size"
    )
    # Use hasattr to check for kernel_size to avoid type issues
    conv_layer = getattr(spatial_attn, "conv", None)
    if conv_layer and hasattr(conv_layer, "kernel_size"):
        assert conv_layer.kernel_size == (5, 5)


def test_cbam_invalid_config_raises():
    """Test invalid CBAM config raises ValueError."""
    dummy = nn.Identity()
    # reduction >= in_channels (invalid)
    config = OmegaConf.create(
        {
            "cbam_enabled": True,
            "cbam_params": {
                "reduction": 4,
                "kernel_size": 3,
                "in_channels": 2,
            },
        }
    )
    with pytest.raises(ValueError):
        insert_cbam_if_enabled(dummy, config)
    # kernel_size par (invalid)
    config = OmegaConf.create(
        {
            "cbam_enabled": True,
            "cbam_params": {
                "reduction": 1,
                "kernel_size": 2,
                "in_channels": 4,
            },
        }
    )
    with pytest.raises(ValueError):
        insert_cbam_if_enabled(dummy, config)
