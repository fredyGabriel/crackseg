import torch
from src.model.decoder.cnn_decoder import DecoderBlock
import pytest


def test_decoderblock_forward_shape():
    in_channels = 16
    skip_channels = 8
    out_channels = 4
    block = DecoderBlock(in_channels, skip_channels, out_channels)

    x = torch.randn(2, in_channels, 16, 16)  # Input from previous layer
    skip = torch.randn(2, skip_channels, 32, 32)  # Skip from encoder

    # Adjust skip connection size to match upsampled x (bilinear does not care)
    # In a real UNet, the skip tensor would have the correct size already.
    # Here we resize for testing the block in isolation.
    skip_resized = torch.nn.functional.interpolate(
        skip, size=x.shape[2]*2, mode='bilinear', align_corners=True
    )

    out = block(x, [skip_resized])

    # Output shape: upsampled and processed
    assert out.shape[0] == x.shape[0]
    assert out.shape[1] == out_channels
    assert out.shape[2] == x.shape[2] * 2  # Upsampled height
    assert out.shape[3] == x.shape[3] * 2  # Upsampled width


def test_decoderblock_properties():
    block = DecoderBlock(10, 5, 3)
    assert block.out_channels == 3


def test_decoderblock_no_skip_error():
    block = DecoderBlock(10, 5, 3)
    x = torch.randn(1, 10, 8, 8)
    with pytest.raises(ValueError, match="one skip connection"):
        block(x, [])  # Empty list


def test_decoderblock_multiple_skips_error():
    block = DecoderBlock(10, 5, 3)
    x = torch.randn(1, 10, 8, 8)
    skip1 = torch.randn(1, 5, 16, 16)
    skip2 = torch.randn(1, 5, 16, 16)
    with pytest.raises(ValueError, match="one skip connection"):
        block(x, [skip1, skip2])  # Multiple skips
