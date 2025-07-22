import logging

import pytest
import torch

from crackseg.model.architectures.cnn_convlstm_unet import CNNEncoder

# NOTE: If you still get argument errors for ASPPModule, check for mocks or
# redefinitions in the test environment or conftest.py.
from crackseg.model.components.aspp import ASPPModule

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "base_filters, in_channels, out_channels, dilations",
    [
        (16, 3, 1, [1, 6, 12, 18]),
    ],
)
def test_simple_aspp(
    base_filters: int,
    in_channels: int,
    out_channels: int,  # Used in parametrize
    dilations: list[int],  # Used in parametrize
):
    """
    Simplified test to verify the basic functionality of ASPP. Checks that
    the ASPP module works correctly independently, without integrating it
    with the entire UNet.
    """
    # Prepare dimensions
    batch_size = 2
    input_height = 64
    input_width = 64

    # Create a simple input
    x = torch.randn(batch_size, in_channels, input_height, input_width)

    # Create the encoder
    encoder = CNNEncoder(
        in_channels=in_channels, base_filters=base_filters, depth=4
    )

    # Extract features and skip connections
    features, _ = encoder(x)

    # Create ASPP module
    aspp = ASPPModule(
        in_channels=encoder.out_channels,
        output_channels=encoder.out_channels,
        dilation_rates=dilations,
    )

    # Apply ASPP
    aspp_output = aspp(features)

    # Check output dimensions
    assert aspp_output.shape[0] == batch_size
    assert aspp_output.shape[1] == encoder.out_channels
    assert aspp_output.shape[2:] == features.shape[2:]

    # Check for NaN or infinite values
    assert torch.isfinite(aspp_output).all()

    logger.info(
        "ASPP test passed: Input shape %s -> Output shape %s",
        features.shape,
        aspp_output.shape,
    )


@pytest.mark.parametrize(
    "base_filters, in_channels, out_channels, dilations",
    [
        (16, 3, 1, [1, 6, 12, 18]),
    ],
)
def test_aspp_simplified_unet(
    base_filters: int,
    in_channels: int,
    out_channels: int,  # Used in parametrize
    dilations: list[int],  # Used in parametrize
):
    """
    Test using a simplified structure for ASPP integration. This test
    verifies that ASPP can be correctly integrated with the UNet decoding
    flow, simulating the key steps: 1. Encoding (encoder) 2. Bottleneck
    (ASPP) 3. First decoding stage
    """
    # Define parameters
    encoder_depth = 4
    input_height = 64
    input_width = 64
    batch_size = 2

    # Create input
    x = torch.randn(batch_size, in_channels, input_height, input_width)

    # 1. Create encoder
    encoder = CNNEncoder(
        in_channels=in_channels, base_filters=base_filters, depth=encoder_depth
    )

    # Apply encoder
    features, skips = encoder(x)

    # 2. Create ASPP bottleneck
    bottleneck = ASPPModule(
        in_channels=encoder.out_channels,
        output_channels=encoder.out_channels,
        dilation_rates=dilations,
    )

    # Apply bottleneck
    bottleneck_output = bottleneck(features)

    # Log channel and dimension info
    logger.info("Encoder out_channels: %s", encoder.out_channels)
    logger.info("Bottleneck out_channels: %s", bottleneck.out_channels)
    logger.info("Encoder skip channels (HIGH->LOW): %s", encoder.skip_channels)

    # Reverse the order of skip_channels for the decoder (LOW->HIGH)
    decoder_skip_channels = list(reversed(encoder.skip_channels))
    logger.info("Decoder skip channels (LOW->HIGH): %s", decoder_skip_channels)

    # 3. Simulate first upsampling stage directly (without adapter)
    # Apply upsampling
    upsampler = torch.nn.Upsample(
        scale_factor=2, mode="bilinear", align_corners=False
    )

    # Upsample the bottleneck output
    upsampled = upsampler(bottleneck_output)

    # Calculate the expected output channels for the first decoder layer
    # This is the same logic used in CNNDecoder to calculate channels
    first_stage_out_channels = encoder.out_channels // 2

    # Create a convolutional layer that performs the appropriate channel
    # reduction
    up_conv = torch.nn.Conv2d(
        encoder.out_channels,
        first_stage_out_channels,
        kernel_size=3,
        padding=1,
    )

    # Apply convolution after upsampling
    upsampled_reduced = up_conv(upsampled)

    # Take the first skip connection (closest to bottleneck/lowest resolution)
    first_skip = skips[-1]

    # Check and adjust spatial dimensions if necessary
    if upsampled_reduced.shape[2:] != first_skip.shape[2:]:
        upsampled_reduced = torch.nn.functional.interpolate(
            upsampled_reduced,
            size=first_skip.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

    logger.info("Upsampled shape: %s", upsampled_reduced.shape)
    logger.info("Skip shape: %s", first_skip.shape)

    # Concatenate with the corresponding skip connection
    concat = torch.cat([upsampled_reduced, first_skip], dim=1)

    # Check dimensions
    expected_concat_channels = (
        first_stage_out_channels + decoder_skip_channels[0]
    )
    assert (
        concat.shape[1] == expected_concat_channels
    ), f"Expected {expected_concat_channels} channels, got {concat.shape[1]}"

    logger.info("Concatenated tensor shape: %s", concat.shape)
    logger.info("Test passed for simplified decoder path")

    # Updated comments on integration:
    # 1. When integrating ASPP into UNet, correctly handle channel dimensions
    # 2. No dynamic adapters needed, calculate channels during initialization
    # 3. Skip connections must be in correct order (LOW->HIGH for CNNDecoder)
