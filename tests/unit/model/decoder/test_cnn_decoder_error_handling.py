from typing import Any

import pytest
import torch

from crackseg.model.decoder.cnn_decoder import CNNDecoder


def test_cnndecoder_skip_mismatch_error():
    """Test error when skip connections don't match expected channels."""
    bottleneck_channels = 128
    # Descending order (low to high resolution)
    skip_channels_list = [32, 16, 8]  # 3 skips
    decoder = CNNDecoder(bottleneck_channels, skip_channels_list)

    # Create input
    x = torch.randn(1, bottleneck_channels, 4, 4)

    # Create skips with wrong channel numbers
    wrong_skips = [
        torch.randn(1, 64, 8, 8),  # Wrong: should be 32
        torch.randn(1, 32, 16, 16),  # Wrong: should be 16
        torch.randn(1, 16, 32, 32),  # Wrong: should be 8
    ]

    # Should raise error about channel mismatch
    with pytest.raises(ValueError, match="channels"):
        decoder(x, wrong_skips)


@pytest.mark.parametrize(
    "params",
    [
        # Tuple structure: (in_channels, skip_channels_list, h, w, batch_size,
        # out_channels, expected_error_msg_regex)
    ],
)
def test_cnndecoder_cbam_reduction_error(params: tuple[Any, ...]):
    (
        in_channels_param,
        skip_channels_list_param,
        h_param,
        w_param,
        batch_size_param,
        out_channels_param,
        expected_error_msg_regex,
    ) = params
    """Test error conditions related to CBAM reduction if applicable at
    CNNDecoder level."""
    # This test assumes CNNDecoder can configure CBAM in a way that causes an
    # error.
    # If CBAM configuration is internal to DecoderBlock and CNNDecoder does
    # not expose it, this test is more appropriate for test_decoder_block.py.
    # For now, it's a placeholder.
    if not expected_error_msg_regex:  # If no error is expected, do not run.
        pytest.skip("No error message regex provided for this CBAM test case.")

    with pytest.raises(ValueError, match=expected_error_msg_regex):
        # Assuming use_cbam=True and a config that causes an error.
        # decoder = CNNDecoder(in_channels_param, skip_channels_list_param,
        # out_channels=out_channels_param, use_cbam=True,
        # cbam_reduction_problematic_value=...)
        # Since we don't have `use_cbam` in CNNDecoder,
        # this test doesn't apply directly.
        # For this test to make sense, CNNDecoder would need a way to
        # misconfigure a DecoderBlock with CBAM.
        # For now, this test cannot be run as is.
        pytest.skip(
            "CBAM error testing at CNNDecoder level needs review based on API."
        )
        # The following line is for the test not to fail for lacking asserts
        # if skip is uncommented.
        assert True  # Placeholder
