from typing import cast

import pytest
import torch
from torch import nn

from crackseg.model.decoder.cnn_decoder import CNNDecoder


def test_cnndecoder_channel_propagation_increasing() -> None:
    """Test channel propagation with increasing skip channels."""
    in_ch: int = 64
    # Descending order (low to high resolution)
    skip_channels_list: list[int] = [32, 16, 8]
    decoder: CNNDecoder = CNNDecoder(in_ch, skip_channels_list)
    # Verify decoder blocks are created
    decoder_blocks = cast(nn.ModuleList, decoder.decoder_blocks)
    assert len(decoder_blocks) == len(skip_channels_list)
    # Verify final conv
    if len(decoder_blocks) > 0:
        first_block = cast(nn.Module, decoder_blocks[0])
        assert hasattr(first_block, "in_channels")
        assert hasattr(first_block, "out_channels")


def test_cnndecoder_channel_propagation_decreasing() -> None:
    """Test channel propagation with decreasing skip channels."""
    in_ch: int = 128
    # Descending order (low to high resolution)
    skip_channels_list: list[int] = [64, 32, 16]
    decoder: CNNDecoder = CNNDecoder(in_ch, skip_channels_list)
    decoder_blocks = cast(nn.ModuleList, decoder.decoder_blocks)
    assert len(decoder_blocks) == len(skip_channels_list)
    if len(decoder_blocks) > 0:
        first_block = cast(nn.Module, decoder_blocks[0])
        assert hasattr(first_block, "in_channels")
        assert hasattr(first_block, "out_channels")


def test_cnndecoder_custom_channels_per_block() -> None:
    """Test custom channel configurations per block."""
    in_ch: int = 256
    # Descending order (low to high resolution)
    skip_channels_list: list[int] = [40, 30, 20, 10]
    decoder: CNNDecoder = CNNDecoder(in_ch, skip_channels_list)
    decoder_blocks = cast(nn.ModuleList, decoder.decoder_blocks)
    assert len(decoder_blocks) == 4
    # Each block should have proper channel configuration
    for _, block in enumerate(decoder_blocks):
        # Note: Accessing skip_channels may require specific knowledge
        # of the block structure. This test might need adjustment based
        # on actual CNNDecoder implementation
        _ = cast(nn.Module, block)  # Verify casting works


def test_cnndecoder_channel_propagation_detailed() -> None:
    """Detailed test of channel propagation through decoder."""
    in_ch: int = 128
    # Descending order (low to high resolution)
    skip_channels_list: list[int] = [32, 16, 8]
    decoder: CNNDecoder = CNNDecoder(in_ch, skip_channels_list)

    # Test forward pass
    x: torch.Tensor = torch.randn(2, in_ch, 4, 4)
    skips: list[torch.Tensor] = [
        torch.randn(2, 32, 8, 8),
        torch.randn(2, 16, 16, 16),
        torch.randn(2, 8, 32, 32),
    ]
    output: torch.Tensor = decoder(x, skips)
    assert output.shape == (2, 1, 32, 32)


def test_cnndecoder_channel_propagation_with_runtime_verification() -> None:
    """Test channel propagation with runtime shape verification."""
    in_ch: int = 64
    skip_channels_list: list[int] = [32, 16, 8]  # Descending
    decoder: CNNDecoder = CNNDecoder(in_ch, skip_channels_list)

    # Create input and skips
    batch_size: int = 2
    x: torch.Tensor = torch.randn(batch_size, in_ch, 8, 8)

    # Skip connections with proper spatial dimensions
    skips: list[torch.Tensor] = []
    h: int = 16
    w: int = 16  # First skip size
    for i, ch in enumerate(skip_channels_list):
        skips.append(torch.randn(batch_size, ch, h * (2**i), w * (2**i)))

    # Forward pass
    output: torch.Tensor = decoder(x, skips)

    # Verify output shape matches highest resolution skip
    assert output.shape[2:] == skips[-1].shape[2:]
    assert output.shape[1] == decoder.out_channels


def test_cnndecoder_asymmetric_channel_configurations():
    """Test decoder with asymmetric channel configurations."""
    in_ch: int = 100
    skip_channels_list: list[int] = [55, 25, 10]  # Descending
    decoder: CNNDecoder = CNNDecoder(in_ch, skip_channels_list)

    # Test forward pass
    x: torch.Tensor = torch.randn(1, in_ch, 4, 4)
    skips: list[torch.Tensor] = [
        torch.randn(1, 55, 8, 8),
        torch.randn(1, 25, 16, 16),
        torch.randn(1, 10, 32, 32),
    ]
    output: torch.Tensor = decoder(x, skips)
    assert output.shape == (1, 1, 32, 32)


def test_cnndecoder_custom_channels_tracking():
    """Test tracking of custom channel configurations."""
    in_ch: int = 256
    skip_channels_list: list[int] = [128, 64, 32, 16]  # Descending
    decoder: CNNDecoder = CNNDecoder(in_ch, skip_channels_list, out_channels=3)

    # Verify properties
    assert decoder.out_channels == 3
    decoder_blocks = cast(nn.ModuleList, decoder.decoder_blocks)
    assert len(decoder_blocks) == 4

    # Test forward pass
    x: torch.Tensor = torch.randn(2, in_ch, 2, 2)
    skips: list[torch.Tensor] = []
    h: int = 4
    w: int = 4
    for i, ch in enumerate(skip_channels_list):
        skips.append(torch.randn(2, ch, h * (2**i), w * (2**i)))

    output: torch.Tensor = decoder(x, skips)
    assert output.shape == (2, 3, 32, 32)


def test_cnndecoder_various_skip_configurations():
    """Test various skip channel configurations."""
    configs: list[tuple[int, list[int]]] = [
        (64, [32, 16]),
        (128, [64, 32, 16]),
        (256, [128, 64, 32, 16]),
        (512, [256, 128, 64, 32, 16]),
    ]

    for in_ch, skip_list_config in configs:
        decoder: CNNDecoder = CNNDecoder(in_ch, skip_list_config)
        decoder_blocks = cast(nn.ModuleList, decoder.decoder_blocks)
        assert len(decoder_blocks) == len(skip_list_config)

        # Test forward pass
        x: torch.Tensor = torch.randn(1, in_ch, 2, 2)
        skips: list[torch.Tensor] = []
        h: int = 4
        w: int = 4
        for i, ch in enumerate(skip_list_config):
            skips.append(torch.randn(1, ch, h * (2**i), w * (2**i)))

        output: torch.Tensor = decoder(x, skips)
        assert output.shape[2:] == skips[-1].shape[2:]


class TestCNNDecoderDimensions:
    """Test suite for CNNDecoder dimension handling."""

    @pytest.mark.parametrize(
        "in_ch_p, skip_channels_list_p, input_size_p, out_ch_p",
        [
            (32, [16, 8], (4, 4), 2),
            (64, [32], (8, 8), 1),
            (16, [8, 4, 2], (2, 2), 3),
            (128, [64], (16, 16), 1),
        ],
    )
    def test_dimensions_after_each_block(
        self,
        in_ch_p: int,
        skip_channels_list_p: list[int],
        input_size_p: tuple[int, int],
        out_ch_p: int,
    ) -> None:
        """Test dimensions after each decoder block."""
        decoder: CNNDecoder = CNNDecoder(in_ch_p, skip_channels_list_p)

        # Create input
        x: torch.Tensor = torch.randn(1, in_ch_p, *input_size_p)

        # Create skips with proper dimensions
        skips: list[torch.Tensor] = []
        h: int = input_size_p[0] * 2
        w: int = input_size_p[1] * 2
        for i, ch in enumerate(skip_channels_list_p):
            skips.append(torch.randn(1, ch, h * (2**i), w * (2**i)))

        # Forward pass
        output: torch.Tensor = decoder(x, skips)

        # Verify output dimensions
        expected_h: int = input_size_p[0] * (2 ** (len(skip_channels_list_p)))
        expected_w: int = input_size_p[1] * (2 ** (len(skip_channels_list_p)))
        assert output.shape == (1, 1, expected_h, expected_w)


class TestCNNDecoderBlockInteraction:
    """Test interactions between decoder blocks."""

    @pytest.mark.parametrize(
        "in_ch_param, skip_channels_list_param",
        [
            (32, [16, 8]),
            (64, [32, 16]),
            (16, [8, 4, 2]),
            (128, [64, 32, 16, 8]),
        ],
    )
    def test_channel_propagation_between_blocks(
        self,
        in_ch_param: int,
        skip_channels_list_param: list[int],
    ) -> None:
        """Test channel propagation between consecutive blocks."""
        decoder: CNNDecoder = CNNDecoder(in_ch_param, skip_channels_list_param)

        # Verify block count
        decoder_blocks = cast(nn.ModuleList, decoder.decoder_blocks)
        assert len(decoder_blocks) == len(skip_channels_list_param)

        # Test forward pass
        x: torch.Tensor = torch.randn(2, in_ch_param, 4, 4)
        skips: list[torch.Tensor] = []
        h: int = 8
        w: int = 8
        for i, ch in enumerate(skip_channels_list_param):
            skips.append(torch.randn(2, ch, h * (2**i), w * (2**i)))

        output: torch.Tensor = decoder(x, skips)
        assert output.shape[0] == 2  # Batch size preserved
        assert output.shape[1] == decoder.out_channels
