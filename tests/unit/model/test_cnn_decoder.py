import torch
from src.model.decoder.cnn_decoder import CNNDecoder, DecoderBlock
import pytest


def test_cnndecoder_init():
    """Test initialization with matching depth and skip channels."""
    in_ch = 128  # Bottleneck out
    skip_ch = [16, 32, 64]  # Encoder skips (high-res to low-res)
    depth = 3
    out_ch = 1

    decoder = CNNDecoder(in_channels=in_ch, skip_channels_list=skip_ch,
                         out_channels=out_ch, depth=depth)

    assert len(decoder.decoder_blocks) == depth
    assert isinstance(decoder.decoder_blocks[0], DecoderBlock)
    # Check channels for the first block (takes bottleneck in, lowest res skip)
    assert decoder.decoder_blocks[0].conv1.in_channels == 80
    assert decoder.decoder_blocks[0].out_channels == skip_ch[-1]  # 64
    # Check channels for the second block
    assert decoder.decoder_blocks[1].conv1.in_channels == 64
    assert decoder.decoder_blocks[1].out_channels == skip_ch[-2]  # 32
    # Check channels for the last block
    assert decoder.decoder_blocks[2].conv1.in_channels == 80
    assert decoder.decoder_blocks[2].out_channels == skip_ch[-3]  # 16

    assert decoder.final_conv.in_channels == skip_ch[-3]  # 16
    assert decoder.final_conv.out_channels == out_ch
    assert decoder.out_channels == out_ch


def test_cnndecoder_forward_shape():
    """Test forward pass output shape."""
    batch_size = 2
    bottleneck_channels = 64
    skip_channels = [8, 16, 32]  # high-res -> low-res
    depth = 3
    out_cls = 5
    H_bottleneck, W_bottleneck = 8, 8

    decoder = CNNDecoder(bottleneck_channels, skip_channels, out_cls, depth)

    # Mock input from bottleneck
    bottleneck_in = torch.randn(batch_size, bottleneck_channels,
                                H_bottleneck, W_bottleneck)

    # Mock skip connections (reverse order needed by CNNDecoder internally)
    # Sizes must match the expected upsampled sizes
    skips_in = []
    current_H, current_W = H_bottleneck, W_bottleneck
    for i in range(depth):
        # low-res skip channel first for test setup
        skip_ch_i = skip_channels[depth - 1 - i]
        current_H *= 2
        current_W *= 2
        skips_in.append(torch.randn(batch_size, skip_ch_i,
                                    current_H, current_W))

    # Skips need to be provided high-res to low-res to forward
    skips_in_correct_order = list(reversed(skips_in))

    output = decoder(bottleneck_in, skips_in_correct_order)

    # Check final output shape
    expected_H_out = H_bottleneck * (2**depth)
    expected_W_out = W_bottleneck * (2**depth)
    assert output.shape == (batch_size, out_cls,
                            expected_H_out, expected_W_out)


def test_cnndecoder_skip_mismatch_error():
    """Test error when number of skips doesn't match depth."""
    with pytest.raises(ValueError, match="Number of skips must match"):
        decoder = CNNDecoder(64, [8, 16, 32], 1, depth=3)
        bottleneck_in = torch.randn(1, 64, 8, 8)
        skips_wrong_num = [torch.randn(1, 8, 16, 16)]  # Only one skip
        decoder(bottleneck_in, skips_wrong_num)


def test_cnndecoder_skip_channels_mismatch_init_error():
    """Test error during init if skip_channels_list length != depth."""
    with pytest.raises(ValueError,
                       match="Length of skip_channels_list must match"):
        # Mismatch: 2 skips provided, but depth=3 expected
        CNNDecoder(64, [8, 16], 1, depth=3)
