import pytest
import torch

from src.model.decoder.cnn_decoder import CNNDecoder


def test_cnndecoder_forward_shape() -> None:
    """Test forward pass output shape."""
    batch_size = 2
    bottleneck_channels = 128
    skip_channels_list = [32, 16, 8]  # Descending (low to high resolution)
    decoder = CNNDecoder(bottleneck_channels, skip_channels_list)
    h_bottleneck, w_bottleneck = 4, 4
    bottleneck_in = torch.randn(
        batch_size, bottleneck_channels, h_bottleneck, w_bottleneck
    )
    skip_tensors = []
    # Increasing spatial dimensions for skips
    h_skip_base, w_skip_base = h_bottleneck * 2, w_bottleneck * 2
    for i, skip_ch in enumerate(skip_channels_list):
        skip_tensors.append(
            torch.randn(
                batch_size, skip_ch, h_skip_base * (2**i), w_skip_base * (2**i)
            )
        )
    output = decoder(bottleneck_in, skip_tensors)
    expected_h, expected_w = skip_tensors[-1].shape[2:]
    assert output.shape == (
        batch_size,
        decoder.out_channels,
        expected_h,
        expected_w,
    )


@pytest.mark.parametrize(
    "bottleneck_ch, skip_channels, out_channels, batch_size",
    [
        (64, [32, 16], 1, 1),
        (128, [64, 32, 16], 3, 2),
        (256, [128, 64], 2, 4),
        (512, [256, 128, 64, 32], 5, 1),
    ],
)
def test_cnndecoder_output_shape_various_configs(
    bottleneck_ch, skip_channels, out_channels, batch_size
):
    """Test output shape with various configurations."""
    decoder = CNNDecoder(
        bottleneck_ch, skip_channels, out_channels=out_channels
    )

    # Create input
    spatial_size = 4
    x = torch.randn(batch_size, bottleneck_ch, spatial_size, spatial_size)

    # Create skips
    skips = []
    for i, ch in enumerate(skip_channels):
        skip_h = spatial_size * (2 ** (i + 1))
        skip_w = spatial_size * (2 ** (i + 1))
        skips.append(torch.randn(batch_size, ch, skip_h, skip_w))

    # Forward pass
    output = decoder(x, skips)

    # Verify output shape
    expected_spatial = spatial_size * (2 ** len(skip_channels))
    assert output.shape == (
        batch_size,
        out_channels,
        expected_spatial,
        expected_spatial,
    )


class CNNDecoderForwardTests:
    """Tests adapted for ascending skips and increasing dimensions."""

    def test_forward_standard_skip_shape_completes(self) -> None:
        in_ch = 64
        skip_list = [8, 16, 32]
        h_in, w_in = 8, 8
        decoder = CNNDecoder(in_ch, skip_list)
        x = torch.randn(2, in_ch, h_in, w_in)
        skips = [
            torch.randn(2, ch, h_in * 2 ** (i + 1), w_in * 2 ** (i + 1))
            for i, ch in enumerate(skip_list)
        ]
        output = decoder(x, skips)
        assert output.shape[1] == decoder.out_channels
        assert output.shape[2:] == skips[-1].shape[2:]

    def test_forward_varied_skip_shapes_completes(self) -> None:
        in_ch = 32
        skip_list = [4, 8, 16, 32]
        h_in, w_in = 4, 4
        decoder = CNNDecoder(in_ch, skip_list)
        x = torch.randn(1, in_ch, h_in, w_in)
        # Skips with diverse increasing spatial dimensions
        skips = [
            torch.randn(1, skip_list[0], h_in * 2, w_in * 2),
            torch.randn(1, skip_list[1], h_in * 4, w_in * 4),
            torch.randn(1, skip_list[2], h_in * 8, w_in * 8),
            torch.randn(1, skip_list[3], h_in * 16, w_in * 16),
        ]
        output = decoder(x, skips)
        assert output.shape[1] == decoder.out_channels
        assert output.shape[2:] == skips[-1].shape[2:]

    def test_forward_edge_case_skip_shapes_completes(self) -> None:
        in_ch = 128
        skip_list = [64]  # Single skip
        h_in, w_in = 16, 16
        decoder = CNNDecoder(in_ch, skip_list)
        x = torch.randn(3, in_ch, h_in, w_in)
        skips = [torch.randn(3, skip_list[0], h_in * 2, w_in * 2)]
        output = decoder(x, skips)
        assert output.shape[1] == decoder.out_channels
        assert output.shape[2:] == skips[-1].shape[2:]


def test_cnndecoder_different_input_sizes() -> None:
    """Test decoder with different input spatial sizes."""
    in_ch = 128
    skip_channels_list = [32, 16, 8]  # Descending
    decoder = CNNDecoder(in_ch, skip_channels_list, out_channels=1)

    input_sizes = [(2, 2), (4, 4), (8, 8), (16, 16)]

    for h_in, w_in in input_sizes:
        x = torch.randn(1, in_ch, h_in, w_in)
        skips = []

        # Create skips with appropriate sizes
        for i, ch in enumerate(skip_channels_list):
            skip_h = h_in * (2 ** (i + 1))
            skip_w = w_in * (2 ** (i + 1))
            skips.append(torch.randn(1, ch, skip_h, skip_w))

        output = decoder(x, skips)

        # Output should be 8x the input size (2^3 for 3 decoder blocks)
        expected_h = h_in * (2 ** len(skip_channels_list))
        expected_w = w_in * (2 ** len(skip_channels_list))
        assert output.shape == (1, 1, expected_h, expected_w)


def test_cnndecoder_segmentation_output_dimensions() -> None:
    """Test decoder output dimensions for segmentation tasks."""
    configs = [
        # (in_ch, skip_channels, num_classes, input_size)
        (128, [64, 32], 2, (8, 8)),  # Binary segmentation
        (256, [128, 64, 32], 5, (4, 4)),  # Multi-class
        (512, [256, 128], 10, (16, 16)),  # Many classes
    ]

    for in_ch, skip_channels, num_classes, input_size in configs:
        decoder = CNNDecoder(in_ch, skip_channels, out_channels=num_classes)

        # Create input
        x = torch.randn(2, in_ch, *input_size)

        # Create skips
        skips = []
        for i, ch in enumerate(skip_channels):
            skip_h = input_size[0] * (2 ** (i + 1))
            skip_w = input_size[1] * (2 ** (i + 1))
            skips.append(torch.randn(2, ch, skip_h, skip_w))

        # Forward pass
        output = decoder(x, skips)

        # Check output
        expected_h = input_size[0] * (2 ** len(skip_channels))
        expected_w = input_size[1] * (2 ** len(skip_channels))
        assert output.shape == (2, num_classes, expected_h, expected_w)

        # Verify output is suitable for loss computation
        assert output.dtype == torch.float32
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


def test_cnndecoder_end_to_end_flow() -> None:
    """Test complete end-to-end flow through decoder."""
    # Simulate bottleneck output
    bottleneck_ch = 256
    bottleneck_h, bottleneck_w = 8, 8

    # Define architecture
    skip_channels = [128, 64, 32, 16]  # Descending order
    num_classes = 3

    # Create decoder
    decoder = CNNDecoder(
        bottleneck_ch, skip_channels, out_channels=num_classes
    )

    # Create inputs
    batch_size = 4
    x = torch.randn(batch_size, bottleneck_ch, bottleneck_h, bottleneck_w)

    # Create skip connections (simulating encoder outputs)
    skips = []
    for i, ch in enumerate(skip_channels):
        skip_h = bottleneck_h * (2 ** (i + 1))
        skip_w = bottleneck_w * (2 ** (i + 1))
        skips.append(torch.randn(batch_size, ch, skip_h, skip_w))

    # Forward pass
    output = decoder(x, skips)

    # Verify output
    final_h = bottleneck_h * (2 ** len(skip_channels))
    final_w = bottleneck_w * (2 ** len(skip_channels))
    assert output.shape == (batch_size, num_classes, final_h, final_w)

    # Test gradient flow
    loss = output.mean()
    loss.backward()

    # Check gradients exist
    for param in decoder.parameters():
        assert param.grad is not None
        assert not torch.isnan(param.grad).any()


@pytest.mark.parametrize(
    "in_ch_param, skip_channels_list_param, input_size_param, "
    "batch_size_param, out_ch_param",
    [
        (32, [16, 8], (4, 4), 2, 1),
        (64, [32], (8, 8), 1, 3),
        (16, [8, 4, 2], (2, 2), 3, 2),
        (128, [64], (16, 16), 1, 4),
    ],
)
def test_final_output_dimensions_parametrized(
    in_ch_param,
    skip_channels_list_param,
    input_size_param,
    batch_size_param,
    out_ch_param,
):
    """Parametrized test for final output dimensions."""
    decoder = CNNDecoder(
        in_ch_param, skip_channels_list_param, out_channels=out_ch_param
    )

    # Create input
    x = torch.randn(batch_size_param, in_ch_param, *input_size_param)

    # Create skips
    skips = []
    h, w = input_size_param[0] * 2, input_size_param[1] * 2
    for i, ch in enumerate(skip_channels_list_param):
        skips.append(torch.randn(batch_size_param, ch, h * (2**i), w * (2**i)))

    # Forward pass
    output = decoder(x, skips)

    # Verify dimensions
    expected_h = input_size_param[0] * (2 ** len(skip_channels_list_param))
    expected_w = input_size_param[1] * (2 ** len(skip_channels_list_param))
    assert output.shape == (
        batch_size_param,
        out_ch_param,
        expected_h,
        expected_w,
    )


@pytest.mark.parametrize(
    "in_ch_param, skip_channels_list_param, input_size_param, "
    "batch_size_param",
    [
        (32, [16, 8], (4, 4), 2),
        (64, [32], (8, 8), 1),
        (16, [8, 4, 2], (2, 2), 3),
        (128, [64], (16, 16), 1),
    ],
)
def test_information_flow_pipeline_parametrized(
    in_ch_param, skip_channels_list_param, input_size_param, batch_size_param
):
    """Test information flow through the decoder pipeline."""
    decoder = CNNDecoder(in_ch_param, skip_channels_list_param)

    # Create input with known pattern
    x = torch.ones(batch_size_param, in_ch_param, *input_size_param) * 0.5

    # Create skips with different patterns
    skips = []
    h, w = input_size_param[0] * 2, input_size_param[1] * 2
    for i, ch in enumerate(skip_channels_list_param):
        # Each skip has a different value to track information flow
        skip_value = (i + 1) * 0.1
        skips.append(
            torch.ones(batch_size_param, ch, h * (2**i), w * (2**i))
            * skip_value
        )

    # Forward pass
    output = decoder(x, skips)

    # Output should be influenced by all inputs
    assert output.shape[0] == batch_size_param
    assert output.shape[1] == decoder.out_channels

    # Check that output has been transformed (not just passthrough)
    assert not torch.allclose(output, torch.ones_like(output) * 0.5)
    assert output.min() != output.max()  # Has variation
