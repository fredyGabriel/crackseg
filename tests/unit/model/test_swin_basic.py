"""Basic test for SwinTransformerEncoder."""

import torch
import pytest
from src.model.encoder.swin_transformer_encoder import SwinTransformerEncoder
from src.model import EncoderBase


def test_swin_transformer_encoder_instantiation():
    """Test basic instantiation of SwinTransformerEncoder."""
    # Create a simple encoder instance
    encoder = SwinTransformerEncoder(
        in_channels=3,
        model_name="swinv2_tiny_window16_256",
        pretrained=False
    )

    # Verify basic properties
    assert isinstance(encoder, EncoderBase)
    assert encoder.in_channels == 3
    assert hasattr(encoder, 'out_channels')
    assert hasattr(encoder, 'skip_channels')

    # Print information for debugging
    print(f"Encoder out_channels: {encoder.out_channels}")
    print(f"Encoder skip_channels: {encoder.skip_channels}")


def test_swin_transformer_encoder_forward():
    """Test forward pass of SwinTransformerEncoder."""
    # Skip if no CUDA available
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for testing")

    # Create encoder
    encoder = SwinTransformerEncoder(
        in_channels=3,
        model_name="swinv2_tiny_window16_256",
        pretrained=False
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
    assert bottleneck.shape[0] == 2  # Batch size preserved
    assert len(skip_connections) > 0  # Should have skip connections
    assert all(s.shape[0] == 2 for s in skip_connections)  # All have batch dim
