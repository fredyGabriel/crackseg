"""Integration tests for SwinTransformerEncoder component with other system
components.

These tests verify that the SwinTransformerEncoder component can be integrated
successfully with other system components, including data pipelines and other
U-Net components.
"""

import os

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from src.model import EncoderBase
from src.model.encoder.swin_transformer_encoder import SwinTransformerEncoder


def load_test_config(config_path: str = None) -> DictConfig:
    """Load Hydra config directly from file path."""
    if config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.abspath(
            os.path.join(
                script_dir,
                "..",
                "..",
                "configs",
                "model",
                "encoder",
                "swin_transformer_encoder.yaml",
            )
        )

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        content = f.read()

    cfg = OmegaConf.create(content)
    return cfg


def test_swin_transformer_integration_with_unet():
    """Test direct integration of SwinTransformerEncoder with UNet manually."""
    # Create encoder directly
    encoder = SwinTransformerEncoder(
        in_channels=3,
        model_name="swinv2_tiny_window16_256",
        pretrained=False,
        features_only=True,
    )

    # Forward pass through encoder
    x = torch.randn(2, 3, 224, 224)
    bottleneck, skip_connections = encoder(x)

    # Verify outputs
    assert bottleneck.shape[0] == 2  # Batch size  # noqa: PLR2004
    assert len(skip_connections) > 0

    # Verify skip_connections property is correct
    assert len(encoder.skip_channels) == len(skip_connections)

    # Output details for verification
    print(f"Encoder out_channels: {encoder.out_channels}")
    print(f"Encoder skip_channels: {encoder.skip_channels}")
    print(f"Bottleneck shape: {bottleneck.shape}")
    for i, skip in enumerate(skip_connections):
        print(f"Skip connection {i} shape: {skip.shape}")

    # Verify encoder can be used in a model
    assert isinstance(encoder, EncoderBase)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Memory test requires CUDA"
)
def test_swin_transformer_memory_usage():
    """Test memory usage of SwinTransformerEncoder."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Create encoder
    encoder = SwinTransformerEncoder(
        in_channels=3, model_name="swinv2_tiny_window16_256", pretrained=False
    )
    encoder = encoder.to("cuda")
    encoder.eval()

    # Create input
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224, device="cuda")

    # Warmup
    with torch.no_grad():
        _ = encoder(x)

    # Memory test
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_memory = torch.cuda.memory_allocated() / (1024 * 1024)

    with torch.no_grad():
        torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        start_time.record()
        bottleneck, skip_connections = encoder(x)
        end_time.record()

        torch.cuda.synchronize()

    inference_time = start_time.elapsed_time(end_time)
    peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
    memory_used = peak_memory - start_memory

    print("\nSwinTransformerEncoder Performance:")
    print(f"  - Inference time: {inference_time:.2f} ms")
    print(f"  - Memory used: {memory_used:.2f} MB")

    assert inference_time > 0
    assert memory_used > 0
