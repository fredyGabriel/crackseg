import pytest
import torch

from src.model.components.cbam import CBAM, ChannelAttention, SpatialAttention


def test_channel_attention_valid_init():
    """Test valid initialization of ChannelAttention."""
    attn = ChannelAttention(in_channels=32, reduction=8)
    assert isinstance(attn, ChannelAttention)
    assert attn.in_channels == 32  # noqa: PLR2004
    assert attn.reduction == 8  # noqa: PLR2004


def test_channel_attention_invalid_reduction():
    """Test ChannelAttention raises error for invalid reduction."""
    with pytest.raises(ValueError):
        ChannelAttention(in_channels=16, reduction=0)
    with pytest.raises(ValueError):
        ChannelAttention(in_channels=16, reduction=16)
    with pytest.raises(ValueError):
        ChannelAttention(in_channels=16, reduction=32)


def test_channel_attention_forward_shape():
    """Test output shape matches input for ChannelAttention."""
    attn = ChannelAttention(in_channels=8, reduction=2)
    x = torch.randn(4, 8, 16, 16)
    out = attn(x)
    assert out.shape == (4, 8, 16, 16)


def test_channel_attention_forward_edge_cases():
    """Test ChannelAttention with batch size 1 and unusual shapes."""
    attn = ChannelAttention(in_channels=4, reduction=2)
    x = torch.randn(1, 4, 7, 5)
    out = attn(x)
    assert out.shape == (1, 4, 7, 5)


def test_channel_attention_grad():
    """Test gradient flow through ChannelAttention."""
    attn = ChannelAttention(in_channels=6, reduction=2)
    x = torch.randn(2, 6, 8, 8, requires_grad=True)
    out = attn(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None


def test_spatial_attention_valid_init():
    """Test valid initialization of SpatialAttention."""
    attn = SpatialAttention(kernel_size=7)
    assert isinstance(attn, SpatialAttention)
    attn = SpatialAttention(kernel_size=3)
    assert isinstance(attn, SpatialAttention)


def test_spatial_attention_invalid_kernel():
    """Test SpatialAttention raises error for invalid kernel size."""
    with pytest.raises(ValueError):
        SpatialAttention(kernel_size=0)
    with pytest.raises(ValueError):
        SpatialAttention(kernel_size=2)


def test_spatial_attention_forward_shape():
    """Test output shape matches input for SpatialAttention."""
    attn = SpatialAttention(kernel_size=7)
    x = torch.randn(3, 10, 12, 12)
    out = attn(x)
    assert out.shape == (3, 10, 12, 12)


def test_spatial_attention_forward_edge_cases():
    """Test SpatialAttention with batch size 1 and unusual shapes."""
    attn = SpatialAttention(kernel_size=3)
    x = torch.randn(1, 5, 5, 9)
    out = attn(x)
    assert out.shape == (1, 5, 5, 9)


def test_spatial_attention_grad():
    """Test gradient flow through SpatialAttention."""
    attn = SpatialAttention(kernel_size=3)
    x = torch.randn(2, 4, 8, 8, requires_grad=True)
    out = attn(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None


def test_cbam_valid_init():
    """Test valid initialization of CBAM."""
    cbam = CBAM(in_channels=8, reduction=2, kernel_size=3)
    assert isinstance(cbam, CBAM)


def test_cbam_forward_shape():
    """Test output shape matches input for CBAM."""
    cbam = CBAM(in_channels=8, reduction=2, kernel_size=3)
    x = torch.randn(2, 8, 16, 16)
    out = cbam(x)
    assert out.shape == (2, 8, 16, 16)


def test_cbam_forward_various_shapes():
    """Test CBAM forward with different input shapes."""
    cbam = CBAM(in_channels=4, reduction=2, kernel_size=3)
    shapes = [(1, 4, 8, 8), (2, 4, 5, 7), (3, 4, 16, 4)]
    for shape in shapes:
        x = torch.randn(*shape)
        out = cbam(x)
        assert out.shape == shape


def test_cbam_grad():
    """Test gradient flow through CBAM."""
    cbam = CBAM(in_channels=6, reduction=2, kernel_size=3)
    x = torch.randn(2, 6, 8, 8, requires_grad=True)
    out = cbam(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
