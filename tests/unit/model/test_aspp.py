from typing import cast

import pytest
import torch
import torch.nn as nn

from crackseg.model.components.aspp import ASPPModule

# Constants (replace with config values if available)
IN_CHANNELS_1 = 32
OUT_CHANNELS_1 = 64
IN_CHANNELS_2 = 16
OUT_CHANNELS_2 = 32
IN_CHANNELS_3 = 8
OUT_CHANNELS_3 = 12
DROPOUT_DEFAULT = 0.1
DROPOUT_HIGH = 0.99
DILATIONS_DEFAULT = [1, 6, 12, 18]
OUTPUT_STRIDE_DEFAULT = 16


def test_asppmodule_valid_initialization():
    """Tests valid initialization of ASPPModule."""
    module = ASPPModule(
        in_channels=IN_CHANNELS_1, output_channels=OUT_CHANNELS_1
    )
    assert isinstance(module, ASPPModule)
    assert isinstance(module, nn.Module)  # Changed from BottleneckBase
    assert module.in_channels == IN_CHANNELS_1
    assert module.out_channels == OUT_CHANNELS_1
    assert module._dilation_rates == DILATIONS_DEFAULT
    assert module._dropout_rate == DROPOUT_DEFAULT
    assert module._output_stride == OUTPUT_STRIDE_DEFAULT


def test_asppmodule_custom_dilations():
    """Tests initialization with custom dilation rates."""
    dilations = [2, 4, 8]
    module = ASPPModule(
        in_channels=IN_CHANNELS_2,
        output_channels=OUT_CHANNELS_2,
        dilation_rates=dilations,
    )
    assert module.in_channels == IN_CHANNELS_2
    assert module.out_channels == OUT_CHANNELS_2
    assert module._dilation_rates == dilations


def test_asppmodule_invalid_channels():
    """Tests that initialization fails with invalid channel dimensions."""
    with pytest.raises(
        ValueError, match="Channel dimensions must be positive"
    ):
        ASPPModule(in_channels=0, output_channels=OUT_CHANNELS_2)
    with pytest.raises(
        ValueError, match="Channel dimensions must be positive"
    ):
        ASPPModule(in_channels=IN_CHANNELS_2, output_channels=0)


def test_asppmodule_empty_dilations():
    """Tests that initialization fails with empty dilation_rates."""
    with pytest.raises(
        ValueError, match="At least one dilation rate must be provided"
    ):
        ASPPModule(
            in_channels=IN_CHANNELS_2,
            output_channels=OUT_CHANNELS_2,
            dilation_rates=[],
        )


def test_asppmodule_invalid_dropout():
    """Tests that initialization fails with invalid dropout rate."""
    with pytest.raises(
        ValueError, match="Dropout rate must be between 0 and 1"
    ):
        ASPPModule(
            in_channels=IN_CHANNELS_2,
            output_channels=OUT_CHANNELS_2,
            dropout_rate=-0.1,
        )
    with pytest.raises(
        ValueError, match="Dropout rate must be between 0 and 1"
    ):
        ASPPModule(
            in_channels=IN_CHANNELS_2,
            output_channels=OUT_CHANNELS_2,
            dropout_rate=1.1,
        )


def test_asppmodule_out_channels_property():
    """Tests the out_channels property."""
    module = ASPPModule(
        in_channels=IN_CHANNELS_3, output_channels=OUT_CHANNELS_3
    )
    assert module.out_channels == OUT_CHANNELS_3


def test_asppmodule_internal_modules():
    """Checks the existence and type of internal ASPP submodules."""
    module = ASPPModule(
        in_channels=IN_CHANNELS_2,
        output_channels=OUT_CHANNELS_2,
        dilation_rates=[1, 2, 3],
    )
    assert hasattr(module, "conv_1x1")
    assert hasattr(module, "branches")
    assert hasattr(module, "global_pool")
    assert hasattr(module, "project")
    assert hasattr(module, "dropout")

    # Check types (basic check)
    assert isinstance(module.conv_1x1, nn.Sequential)
    assert isinstance(module.branches, nn.ModuleList)
    assert isinstance(module.global_pool, nn.Sequential)
    assert isinstance(module.project, nn.Sequential)

    # Use safe attribute access to help type checker
    dropout_rate = getattr(module, "_dropout_rate", 0.0)
    if dropout_rate > 0:
        assert isinstance(module.dropout, nn.Dropout2d)
    else:
        assert isinstance(module.dropout, nn.Identity)


def test_asppmodule_branch_count_matches_dilations():
    """Ensures the number of atrous branches matches dilation rates."""
    dilations = [1, 2, 4, 8]
    module = ASPPModule(
        in_channels=IN_CHANNELS_3,
        output_channels=IN_CHANNELS_3,
        dilation_rates=dilations,
    )
    branches = cast(nn.ModuleList, module.branches)
    assert len(branches) == len(dilations)


def test_asppmodule_forward_output_shape():
    """Tests the output shape of the forward pass."""
    module = ASPPModule(
        in_channels=IN_CHANNELS_3, output_channels=OUT_CHANNELS_2
    )
    input_tensor = torch.randn(2, IN_CHANNELS_3, 32, 32)  # B, C, H, W
    output = module(input_tensor)
    assert output.shape == (2, OUT_CHANNELS_2, 32, 32)  # B, Cout, H, W


def test_asppmodule_forward_various_shapes():
    """Tests forward pass with different input H/W."""
    module = ASPPModule(in_channels=4, output_channels=4)
    module.eval()
    shapes = [(16, 16), (64, 32), (128, 128)]
    for h, w in shapes:
        input_tensor = torch.randn(1, 4, h, w)
        with torch.no_grad():
            output = module(input_tensor)
        assert output.shape == (1, 4, h, w)
        assert torch.isfinite(output).all()


def test_asppmodule_forward_dropout_training():
    """Tests that dropout is applied during training."""
    module = ASPPModule(
        in_channels=4, output_channels=4, dropout_rate=DROPOUT_HIGH
    )
    module.train()  # Set to training mode
    input_tensor = torch.ones(10, 4, 8, 8) * 1000  # Use large values
    output = module(input_tensor)
    # With high dropout, likely many zeros
    assert (output == 0).any()


def test_asppmodule_forward_dropout_eval():
    """Tests that dropout is NOT applied during evaluation."""
    module = ASPPModule(
        in_channels=4, output_channels=4, dropout_rate=DROPOUT_HIGH
    )
    module.eval()  # Set to evaluation mode
    input_tensor = torch.ones(10, 4, 8, 8) * 1000
    output = module(input_tensor)
    # Dropout should not be applied, no zeros expected (unless conv hits 0)
    # Check that it's not *all* zeros, which is highly unlikely without dropout
    assert not (output == 0).all()
    # A stronger check: output should not be drastically scaled down or zeroed
    # compared to input (but exact values depend on learned weights)
    assert output.abs().mean() > 1  # Heuristic: mean should be > 1
