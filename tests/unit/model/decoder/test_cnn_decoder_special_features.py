import pytest
import torch

from src.model.decoder.cnn_decoder import CNNDecoder, CNNDecoderConfig


def test_cnndecoder_custom_target_size():
    """Test CNNDecoder with custom target size (if supported)."""
    # Note: Current implementation may not support target_size
    # This test documents expected behavior
    in_ch = 128
    skip_channels_list = [32, 16, 8]  # Descending

    # Create decoder without target_size (default behavior)
    decoder = CNNDecoder(in_ch, skip_channels_list)

    # Test forward pass
    x = torch.randn(1, in_ch, 4, 4)
    skips = [
        torch.randn(1, 32, 8, 8),
        torch.randn(1, 16, 16, 16),
        torch.randn(1, 8, 32, 32),
    ]

    output = decoder(x, skips)

    # Output size should match highest resolution skip
    assert output.shape[2:] == skips[-1].shape[2:]

    # If target_size is implemented in future:
    # decoder_with_target = CNNDecoder(
    #     in_ch, skip_channels_list, target_size=(64, 64)
    # )
    # output_targeted = decoder_with_target(x, skips)
    # assert output_targeted.shape[2:] == (64, 64)


@pytest.mark.parametrize(
    "in_ch, skip_channels_list_test, depth_test, batch_size, h_in, w_in",
    [
        (64, [32, 16, 8], 3, 4, 16, 16),
        (32, [16, 8], 2, 2, 8, 12),
    ],
)
def test_cnndecoder_forward_cbam_and_shapes(
    in_ch: int,
    skip_channels_list_test: list[int],
    depth_test: int,
    batch_size: int,
    h_in: int,
    w_in: int,
):
    """Test CNNDecoder with CBAM enabled and various shapes."""
    # Create config with CBAM enabled
    config = CNNDecoderConfig(use_cbam=True, cbam_reduction=8)

    decoder = CNNDecoder(
        in_ch,
        skip_channels_list_test,
        out_channels=1,
        depth=depth_test,
        config=config,
    )

    # Create input
    x = torch.randn(batch_size, in_ch, h_in, w_in)

    # Create skips with proper dimensions
    skips = []
    for i, ch in enumerate(skip_channels_list_test):
        skip_h = h_in * (2 ** (i + 1))
        skip_w = w_in * (2 ** (i + 1))
        skips.append(torch.randn(batch_size, ch, skip_h, skip_w))

    # Forward pass
    output = decoder(x, skips)

    # Verify output shape
    expected_h = h_in * (2**depth_test)
    expected_w = w_in * (2**depth_test)
    assert output.shape == (batch_size, 1, expected_h, expected_w)

    # Verify CBAM is being used (check for CBAM modules in decoder blocks)
    has_cbam = any(
        hasattr(block, "cbam") and block.cbam.__class__.__name__ != "Identity"
        for block in decoder.decoder_blocks
    )
    assert has_cbam, "CBAM should be present in decoder blocks"
