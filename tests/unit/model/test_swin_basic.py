"""Basic test for SwinTransformerEncoder."""

import pytest
import torch

from crackseg.model import EncoderBase
from crackseg.model.encoder.swin_transformer_encoder import (
    SwinTransformerEncoder,
    SwinTransformerEncoderConfig,
)


def test_swin_transformer_encoder_instantiation():
    """Test basic instantiation of SwinTransformerEncoder."""
    # Create a simple encoder instance
    config = SwinTransformerEncoderConfig(
        model_name="swinv2_tiny_window16_256",
        pretrained=False,
    )
    encoder = SwinTransformerEncoder(
        in_channels=3,
        config=config,
    )

    # Verify basic properties
    assert isinstance(encoder, EncoderBase)
    assert encoder.in_channels == 3  # noqa: PLR2004
    assert hasattr(encoder, "out_channels")
    assert hasattr(encoder, "skip_channels")

    # Print information for debugging
    print(f"Encoder out_channels: {encoder.out_channels}")
    print(f"Encoder skip_channels: {encoder.skip_channels}")


def test_swin_transformer_encoder_forward():
    """Test forward pass of SwinTransformerEncoder."""
    # Skip if no CUDA available
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for testing")

    # Create encoder
    config = SwinTransformerEncoderConfig(
        model_name="swinv2_tiny_window16_256",
        pretrained=False,
    )
    encoder = SwinTransformerEncoder(
        in_channels=3,
        config=config,
    )

    # Move to CUDA for testing
    encoder = encoder.to("cuda")
    encoder.eval()

    # Create test input
    x = torch.randn(2, 3, 224, 224, device="cuda")

    # Forward pass
    with torch.no_grad():
        bottleneck, skip_connections = encoder(x)

    # Verify output shapes
    print(f"Bottleneck shape: {bottleneck.shape}")
    print(f"Skip connection shapes: {[s.shape for s in skip_connections]}")

    # Basic assertions
    assert bottleneck.shape[0] == 2  # Batch size preserved  # noqa: PLR2004
    assert len(skip_connections) > 0  # Should have skip connections
    assert all(s.shape[0] == 2 for s in skip_connections)  # noqa: PLR2004
