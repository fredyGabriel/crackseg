"""Unit tests for the BaseUNet model implementation."""

import pytest
import torch
from torch import nn

from crackseg.model import (
    BaseUNet,
    BottleneckBase,
    DecoderBase,
    EncoderBase,
)
from tests.unit.model.test_registry import (
    MockBottleneck,
    MockDecoder,
    MockEncoder,
)


class TestBaseUNet:
    """Test cases for the BaseUNet model implementation."""

    @pytest.fixture
    def encoder(self) -> MockEncoder:
        """Create a mock encoder for testing."""
        return MockEncoder(in_channels=3)

    @pytest.fixture
    def bottleneck(self, encoder: MockEncoder) -> MockBottleneck:
        """Create a mock bottleneck for testing."""
        return MockBottleneck(in_channels=encoder.out_channels)

    @pytest.fixture
    def decoder(
        self, bottleneck: MockBottleneck, encoder: MockEncoder
    ) -> MockDecoder:
        """
        Create a mock decoder for testing.

        Following the UNet contract, the decoder must receive the skip_channels
        in reverse order from the encoder (low->high resolution).
        """
        return MockDecoder(
            in_channels=bottleneck.out_channels,
            skip_channels=list(reversed(encoder.skip_channels)),
        )

    @pytest.fixture
    def base_unet(
        self,
        encoder: MockEncoder,
        bottleneck: MockBottleneck,
        decoder: MockDecoder,
    ) -> BaseUNet:
        """Create a BaseUNet instance for testing."""
        return BaseUNet(encoder, bottleneck, decoder)

    @pytest.fixture
    def base_unet_with_activation(
        self,
        encoder: MockEncoder,
        bottleneck: MockBottleneck,
        decoder: MockDecoder,
    ) -> BaseUNet:
        """Create a BaseUNet instance with final activation for testing."""
        return BaseUNet(
            encoder, bottleneck, decoder, final_activation=nn.Sigmoid()
        )

    def test_initialization(self, base_unet: BaseUNet) -> None:
        """Test that the BaseUNet initializes correctly."""
        assert isinstance(base_unet, BaseUNet)
        assert isinstance(base_unet.encoder, EncoderBase)
        assert isinstance(base_unet.bottleneck, BottleneckBase)
        assert isinstance(base_unet.decoder, DecoderBase)
        assert base_unet.final_activation is None

    def test_initialization_with_activation(
        self, base_unet_with_activation: BaseUNet
    ) -> None:
        """Test initialization with final activation."""
        assert isinstance(
            base_unet_with_activation.final_activation, nn.Sigmoid
        )

    def test_get_input_channels(
        self, base_unet: BaseUNet, encoder: MockEncoder
    ) -> None:
        """Test get_input_channels method."""
        assert base_unet.get_input_channels() == encoder.in_channels

    def test_get_output_channels(
        self, base_unet: BaseUNet, decoder: MockDecoder
    ) -> None:
        """Test get_output_channels method."""
        assert base_unet.get_output_channels() == decoder.out_channels

    def test_forward_pass(self, base_unet: BaseUNet) -> None:
        """Test the forward pass through the model."""
        # Create a test input tensor - batch of 2, 3 channels, 64x64
        x = torch.randn(2, base_unet.get_input_channels(), 64, 64)
        output = base_unet(x)

        # Our MockDecoder's forward just passes through without changing shape,
        # so the output matches the input shape
        assert output.shape == x.shape

    def test_forward_with_activation(
        self, base_unet_with_activation: BaseUNet
    ) -> None:
        """Test forward pass with activation."""
        # Create a test input tensor
        x = torch.randn(
            2, base_unet_with_activation.get_input_channels(), 64, 64
        )
        output = base_unet_with_activation(x)

        # Shape should be the same as input since our mock just passes through
        assert output.shape == x.shape
        # Check the output is in the range [0, 1] due to sigmoid
        assert torch.all(output >= 0) and torch.all(output <= 1)

    def test_summary(self, base_unet: BaseUNet) -> None:
        """Test the summary method."""
        summary = base_unet.summary()

        # Ensure components are not None before accessing properties
        assert base_unet.encoder is not None
        assert base_unet.decoder is not None
        assert base_unet.bottleneck is not None

        # Basic info
        assert summary["model_type"] == "BaseUNet"
        assert summary["input_channels"] == base_unet.encoder.in_channels
        assert summary["output_channels"] == base_unet.decoder.out_channels
        assert summary["encoder_type"] == base_unet.encoder.__class__.__name__
        assert (
            summary["bottleneck_type"]
            == base_unet.bottleneck.__class__.__name__
        )
        assert summary["decoder_type"] == base_unet.decoder.__class__.__name__
        assert summary["has_final_activation"] is False
        assert summary["final_activation_type"] is None

        # Parameter counts - mocks might not have real parameters
        assert "parameters" in summary
        assert "total" in summary["parameters"]
        assert "trainable" in summary["parameters"]
        assert "non_trainable" in summary["parameters"]
        # Don't assert specifically on parameter counts as mocks might not
        # have any

        # Memory and receptive field
        assert "memory_usage" in summary
        assert "model_size_mb" in summary["memory_usage"]
        assert "receptive_field" in summary

        # Layer hierarchy
        assert "layer_hierarchy" in summary
        # At least encoder, bottleneck, decoder
        assert len(summary["layer_hierarchy"]) >= 3  # noqa: PLR2004

        # Test memory estimates with input shape
        detailed_summary = base_unet.summary(input_shape=(1, 3, 256, 256))
        assert "estimated_activation_mb" in detailed_summary["memory_usage"]
        assert "total_estimated_mb" in detailed_summary["memory_usage"]

    def test_summary_with_activation(
        self, base_unet_with_activation: BaseUNet
    ) -> None:
        """Test the summary method with activation."""
        summary = base_unet_with_activation.summary()

        # With our fix, the activation should be properly detected
        assert summary["has_final_activation"] is True
        assert summary["final_activation_type"] == "Sigmoid"

        # Should have one more layer in hierarchy for activation
        # encoder, bottleneck, decoder, activation
        assert len(summary["layer_hierarchy"]) >= 4  # noqa: PLR2004

    def test_print_summary(self, base_unet: BaseUNet) -> None:
        """Test the print_summary method (return as string)."""
        summary_str = base_unet.print_summary(
            input_shape=(1, 3, 256, 256), return_string=True
        )

        # Basic checks on the returned string
        assert isinstance(summary_str, str)
        assert "U-Net Model Summary" in summary_str
        assert "Total Parameters:" in summary_str
        assert "Trainable Parameters:" in summary_str
        assert "Model Size:" in summary_str
        assert "Layer Hierarchy:" in summary_str

        # Check for specific sections and formatting
        assert "=" * 80 in summary_str  # Section separators
        assert "-" * 80 in summary_str  # Table separators

        # Should contain all major components
        assert "Encoder" in summary_str
        assert "Bottleneck" in summary_str
        assert "Decoder" in summary_str

    def test_component_compatibility(
        self,
        encoder: MockEncoder,
        bottleneck: MockBottleneck,
        decoder: MockDecoder,
    ) -> None:
        """Test that component compatibility is correctly validated."""
        # This should work fine with compatible components
        BaseUNet(encoder, bottleneck, decoder)

        # Create incompatible decoder (different skip channels)
        incompatible_decoder = MockDecoder(
            in_channels=bottleneck.out_channels,
            skip_channels=[8, 16],  # Different from encoder.skip_channels
        )

        # Should raise an error due to incompatible skip channels
        with pytest.raises(ValueError, match="Encoder skip channels"):
            BaseUNet(encoder, bottleneck, incompatible_decoder)

        # Create incompatible bottleneck (different in/out channels)
        # Not matching encoder.out_channels
        incompatible_bottleneck = MockBottleneck(in_channels=32)

        # Should raise an error due to incompatible channels
        with pytest.raises(ValueError, match="Encoder output channels"):
            BaseUNet(encoder, incompatible_bottleneck, decoder)
