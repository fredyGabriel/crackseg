import torch
from src.model.encoder.cnn_encoder import EncoderBlock


def test_encoderblock_forward_and_skips():
    in_channels = 3
    out_channels = 8
    block = EncoderBlock(in_channels, out_channels, use_pool=True)
    x = torch.randn(2, in_channels, 32, 32)
    out, skips = block(x)
    # Output shape: pooled, skips shape: pre-pooled
    assert out.shape[0] == x.shape[0]
    assert out.shape[1] == out_channels
    assert out.shape[2] == x.shape[2] // 2  # pooled
    assert out.shape[3] == x.shape[3] // 2
    assert isinstance(skips, list)
    assert skips[0].shape == (2, out_channels, 32, 32)


def test_encoderblock_no_pool():
    in_channels = 4
    out_channels = 16
    block = EncoderBlock(in_channels, out_channels, use_pool=False)
    x = torch.randn(1, in_channels, 16, 16)
    out, skips = block(x)
    # Output shape: no pooling
    assert out.shape == (1, out_channels, 16, 16)
    assert skips[0].shape == (1, out_channels, 16, 16)


def test_encoderblock_properties():
    block = EncoderBlock(2, 5)
    assert block.out_channels == 5
    assert block.skip_channels == [5]
