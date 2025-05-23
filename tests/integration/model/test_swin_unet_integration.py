"""
Integration test for Swin-UNet architecture
(SwinTransformerEncoder + CNNDecoder).

This test validates the Swin encoder and the CNN decoder separately.
"""

import torch

from src.model.decoder.cnn_decoder import CNNDecoder
from src.model.encoder.swin_transformer_encoder import (
    SwinTransformerEncoder,
    SwinTransformerEncoderConfig,
)


def test_swin_encoder():
    """Validate that SwinTransformerEncoder works correctly."""
    batch_size = 2
    in_channels = 3
    height = width = 224

    # Configure the encoder
    config = SwinTransformerEncoderConfig(
        model_name="swinv2_tiny_window16_256",
        pretrained=False,
        features_only=True,
    )
    encoder = SwinTransformerEncoder(
        in_channels=in_channels,
        config=config,
    )

    # Forward pass
    x = torch.randn(batch_size, in_channels, height, width)
    with torch.no_grad():
        features, skips = encoder(x)

    # Validations
    assert isinstance(features, torch.Tensor)
    assert isinstance(skips, list)
    assert len(skips) > 0
    assert all(isinstance(skip, torch.Tensor) for skip in skips)

    # Debug log
    print(f"Features shape: {features.shape}")
    for i, skip in enumerate(skips):
        print(f"Skip {i} shape: {skip.shape}")
    print(f"Skip channels from encoder.skip_channels: {encoder.skip_channels}")
    real_skip_channels = [s.shape[1] for s in skips]
    print(f"Skip channels from tensor shapes: {real_skip_channels}")

    # Check that skip_channels matches the real skip tensor channels
    assert (
        encoder.skip_channels == real_skip_channels
    ), "skip_channels property should match actual skip tensor channels"


@torch.no_grad()
def test_vanilla_cnn_decoder():
    """Simple test of CNN Decoder with known channels."""
    # Only check that the decoder initializes without errors
    in_channels = 128
    skip_channels = [64, 32, 16]  # Format low->high
    out_channels = 1

    # Create the decoder with compatible channels
    decoder = CNNDecoder(
        in_channels=in_channels,
        skip_channels_list=skip_channels,
        out_channels=out_channels,
        depth=len(skip_channels),
    )

    # Print channels for debug
    print(f"Decoder in_channels: {in_channels}")
    print(f"Decoder skip_channels: {decoder.skip_channels}")

    # Check properties
    assert decoder.in_channels == in_channels
    assert decoder.skip_channels == skip_channels
    assert decoder.out_channels == out_channels
