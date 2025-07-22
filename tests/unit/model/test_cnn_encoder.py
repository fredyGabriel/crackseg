# ruff: noqa: PLR2004
from typing import cast

import torch
from torch import nn

from crackseg.model.encoder.cnn_encoder import CNNEncoder, EncoderBlock


def test_cnnencoder_init():
    """Test initialization with different depths."""
    encoder_d3 = CNNEncoder(in_channels=3, init_features=16, depth=3)
    encoder_blocks = cast(nn.ModuleList, encoder_d3.encoder_blocks)
    assert len(encoder_blocks) == 3

    # Cast individual blocks to EncoderBlock type
    block_0 = cast(EncoderBlock, encoder_blocks[0])
    block_1 = cast(EncoderBlock, encoder_blocks[1])
    block_2 = cast(EncoderBlock, encoder_blocks[2])

    assert isinstance(block_0, EncoderBlock)
    assert hasattr(block_0, "conv1") and block_0.conv1.in_channels == 3
    assert hasattr(block_0, "conv1") and block_0.conv1.out_channels == 16
    assert hasattr(block_1, "conv1") and block_1.conv1.in_channels == 16
    assert hasattr(block_1, "conv1") and block_1.conv1.out_channels == 32
    assert hasattr(block_2, "conv1") and block_2.conv1.in_channels == 32
    assert hasattr(block_2, "conv1") and block_2.conv1.out_channels == 64
    assert encoder_d3.out_channels == 64  # Channels before last pool
    assert encoder_d3.skip_channels == [16, 32, 64]

    encoder_d1 = CNNEncoder(in_channels=1, init_features=8, depth=1)
    encoder_blocks_d1 = cast(nn.ModuleList, encoder_d1.encoder_blocks)
    assert len(encoder_blocks_d1) == 1
    assert encoder_d1.out_channels == 8
    assert encoder_d1.skip_channels == [8]


def test_cnnencoder_forward_shape():
    """Test forward pass output and skip shapes."""
    batch_size = 2
    in_channels = 3
    init_features = 8
    depth = 3
    H, W = 64, 64
    encoder = CNNEncoder(in_channels, init_features, depth)

    x = torch.randn(batch_size, in_channels, H, W)
    final_out, skips = encoder(x)

    # Check final output shape (after depth=3 blocks and pools)
    expected_out_channels = init_features * (2 ** (depth - 1))
    expected_H = H // (2**depth)
    expected_W = W // (2**depth)
    assert final_out.shape == (
        batch_size,
        expected_out_channels,
        expected_H,
        expected_W,
    )

    # Check skip connections shapes (from high-res to low-res)
    assert len(skips) == depth
    expected_skip_channels = [init_features * (2**i) for i in range(depth)]
    expected_skip_H = [H // (2**i) for i in range(depth)]
    expected_skip_W = [W // (2**i) for i in range(depth)]

    for i in range(depth):
        assert skips[i].shape == (
            batch_size,
            expected_skip_channels[i],
            expected_skip_H[i],
            expected_skip_W[i],
        )


def test_cnnencoder_properties():
    """Test the out_channels and skip_channels properties."""
    encoder = CNNEncoder(in_channels=3, init_features=16, depth=4)
    assert encoder.out_channels == 16 * (2 ** (4 - 1))  # 128
    assert encoder.skip_channels == [16, 32, 64, 128]
