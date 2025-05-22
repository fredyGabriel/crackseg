import torch

from src.model.bottleneck.cnn_bottleneck import BottleneckBlock


def test_bottleneckblock_forward_shape():
    in_channels = 8
    out_channels = 16
    block = BottleneckBlock(in_channels, out_channels)
    x = torch.randn(2, in_channels, 16, 16)
    out = block(x)
    # Output shape: mantiene dimensiones espaciales
    assert out.shape == (2, out_channels, 16, 16)


def test_bottleneckblock_dropout_effect():
    block = BottleneckBlock(4, 4, dropout=1.0)  # Dropout extremo
    x = torch.randn(1, 4, 8, 8)
    block.train()
    out = block(x)
    # No assertion estricta, pero debe ejecutarse sin error y shape igual
    assert out.shape == (1, 4, 8, 8)


def test_bottleneckblock_properties():
    block = BottleneckBlock(3, 7)
    assert block.out_channels == 7  # noqa: PLR2004
