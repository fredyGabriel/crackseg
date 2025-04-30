"""Unit tests for the BaseUNet model implementation."""

import pytest
import torch
import torch.nn as nn

from src.model.base import EncoderBase, BottleneckBase, DecoderBase
from src.model.unet import BaseUNet
from tests.model.test_registry import (
    MockEncoder, MockBottleneck, MockDecoder
)


class TestBaseUNet:
    """Test cases for the BaseUNet model implementation."""

    @pytest.fixture
    def encoder(self):
        """Create a mock encoder for testing."""
        return MockEncoder(in_channels=3)

    @pytest.fixture
    def bottleneck(self, encoder):
        """Create a mock bottleneck for testing."""
        return MockBottleneck(in_channels=encoder.out_channels)

    @pytest.fixture
    def decoder(self, bottleneck, encoder):
        """Create a mock decoder for testing."""
        return MockDecoder(
            in_channels=bottleneck.out_channels,
            skip_channels=encoder.skip_channels
        )

    @pytest.fixture
    def base_unet(self, encoder, bottleneck, decoder):
        """Create a BaseUNet instance for testing."""
        return BaseUNet(encoder, bottleneck, decoder)

    @pytest.fixture
    def base_unet_with_activation(self, encoder, bottleneck, decoder):
        """Create a BaseUNet instance with final activation for testing."""
        return BaseUNet(
            encoder, bottleneck, decoder,
            final_activation=nn.Sigmoid()
        )

    def test_initialization(self, base_unet):
        """Test that the BaseUNet initializes correctly."""
        assert isinstance(base_unet, BaseUNet)
        assert isinstance(base_unet.encoder, EncoderBase)
        assert isinstance(base_unet.bottleneck, BottleneckBase)
        assert isinstance(base_unet.decoder, DecoderBase)
        assert base_unet.final_activation is None

    def test_initialization_with_activation(self, base_unet_with_activation):
        """Test that the BaseUNet initializes correctly with activation."""
        assert isinstance(base_unet_with_activation, BaseUNet)
        assert isinstance(base_unet_with_activation.final_activation,
                          nn.Sigmoid)

    def test_forward_pass(self, base_unet):
        """Test the forward pass through the BaseUNet."""
        # Create a sample input tensor
        x = torch.randn(2, 3, 64, 64)  # Batch of 2, 3 channels, 64x64 image

        # Run forward pass
        output = base_unet(x)

        # Check output shape (MockDecoder's forward just returns x without
        # changing shape)
        assert output.shape == x.shape

        # Note: In a real implementation, output channels would match
        # decoder.out_channels but our mock implementation doesn't modify the
        # tensor shape

    def test_forward_pass_with_activation(self, base_unet_with_activation):
        """Test the forward pass with final activation."""
        # Create a sample input tensor
        x = torch.randn(2, 3, 64, 64)

        # Run forward pass
        output = base_unet_with_activation(x)

        # Check output shape (MockDecoder's forward just returns x without
        # changing shape)
        assert output.shape == x.shape

        # Check that sigmoid was applied (all values between 0 and 1)
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)

    def test_get_input_channels(self, base_unet, encoder):
        """Test the get_input_channels method."""
        assert base_unet.get_input_channels() == encoder.in_channels

    def test_get_output_channels(self, base_unet, decoder):
        """Test the get_output_channels method."""
        assert base_unet.get_output_channels() == decoder.out_channels

    def test_summary(self, base_unet):
        """Test the summary method."""
        summary = base_unet.summary()

        assert summary["model_type"] == "BaseUNet"
        assert summary["input_channels"] == base_unet.encoder.in_channels
        assert summary["output_channels"] == base_unet.decoder.out_channels
        assert summary["encoder_type"] == base_unet.encoder.__class__.__name__
        assert summary["bottleneck_type"] ==\
            base_unet.bottleneck.__class__.__name__
        assert summary["decoder_type"] == base_unet.decoder.__class__.__name__
        assert summary["has_final_activation"] is False
        assert summary["final_activation_type"] is None

    def test_summary_with_activation(self, base_unet_with_activation):
        """Test the summary method with activation."""
        summary = base_unet_with_activation.summary()

        assert summary["has_final_activation"] is True
        assert summary["final_activation_type"] == "Sigmoid"

    def test_component_compatibility(self, encoder, bottleneck, decoder):
        """Test that component compatibility is correctly validated."""
        # This should work fine with compatible components
        BaseUNet(encoder, bottleneck, decoder)

        # Create incompatible decoder (different skip channels)
        incompatible_decoder = MockDecoder(
            in_channels=bottleneck.out_channels,
            skip_channels=[8, 16]  # Different from encoder.skip_channels
        )

        # Should raise an error due to incompatible skip channels
        with pytest.raises(ValueError, match="Encoder skip channels"):
            BaseUNet(encoder, bottleneck, incompatible_decoder)
