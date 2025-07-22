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
