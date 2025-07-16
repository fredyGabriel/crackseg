from typing import cast

import pytest
import torch

from crackseg.model.decoder.cnn_decoder import CNNDecoder


def test_cnndecoder_forward_shape() -> None:
    """Test forward pass output shape."""
    batch_size: int = 2
    bottleneck_channels: int = 128
    skip_channels_list: list[int] = [
        32,
        16,
        8,
    ]  # Descending (low to high resolution)
    decoder = CNNDecoder(bottleneck_channels, skip_channels_list)
    h_bottleneck: int = 4
    w_bottleneck: int = 4
    bottleneck_in: torch.Tensor = torch.randn(
        batch_size, bottleneck_channels, h_bottleneck, w_bottleneck
    )
    skip_tensors: list[torch.Tensor] = []
    # Increasing spatial dimensions for skips
    h_skip_base: int = h_bottleneck * 2
    w_skip_base: int = w_bottleneck * 2
    for i, skip_ch in enumerate(skip_channels_list):
        skip_tensors.append(
            torch.randn(
                batch_size, skip_ch, h_skip_base * (2**i), w_skip_base * (2**i)
            )
        )
    output: torch.Tensor = decoder(bottleneck_in, skip_tensors)
    expected_h: int
    expected_w: int
    expected_h, expected_w = cast(tuple[int, int], skip_tensors[-1].shape[2:])
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
    bottleneck_ch: int,
    skip_channels: list[int],
    out_channels: int,
    batch_size: int,
) -> None:
    """Test output shape with various configurations."""
    decoder = CNNDecoder(
        bottleneck_ch, skip_channels, out_channels=out_channels
    )

    # Create input
    spatial_size: int = 4
    x: torch.Tensor = torch.randn(
        batch_size, bottleneck_ch, spatial_size, spatial_size
    )

    # Create skips
    skips: list[torch.Tensor] = []
    for i, ch in enumerate(skip_channels):
        skip_h: int = spatial_size * (2 ** (i + 1))
        skip_w: int = spatial_size * (2 ** (i + 1))
        skips.append(torch.randn(batch_size, ch, skip_h, skip_w))

    # Forward pass
    output: torch.Tensor = decoder(x, skips)

    # Verify output shape
    expected_spatial: int = spatial_size * (2 ** len(skip_channels))
    assert output.shape == (
        batch_size,
        out_channels,
        expected_spatial,
        expected_spatial,
    )


class CNNDecoderForwardTests:
    """Tests adapted for ascending skips and increasing dimensions."""

    def test_forward_standard_skip_shape_completes(self) -> None:
        in_ch: int = 64
        skip_list: list[int] = [8, 16, 32]
        h_in: int = 8
        w_in: int = 8
        decoder = CNNDecoder(in_ch, skip_list)
        x: torch.Tensor = torch.randn(2, in_ch, h_in, w_in)
        skips: list[torch.Tensor] = [
            torch.randn(2, ch, h_in * 2 ** (i + 1), w_in * 2 ** (i + 1))
            for i, ch in enumerate(skip_list)
        ]
        output: torch.Tensor = decoder(x, skips)
        assert output.shape[1] == decoder.out_channels
        assert output.shape[2:] == skips[-1].shape[2:]

    def test_forward_varied_skip_shapes_completes(self) -> None:
        in_ch: int = 32
        skip_list: list[int] = [4, 8, 16, 32]
        h_in: int = 4
        w_in: int = 4
        decoder = CNNDecoder(in_ch, skip_list)
        x: torch.Tensor = torch.randn(1, in_ch, h_in, w_in)
        # Skips with diverse increasing spatial dimensions
        skips: list[torch.Tensor] = [
            torch.randn(1, skip_list[0], h_in * 2, w_in * 2),
            torch.randn(1, skip_list[1], h_in * 4, w_in * 4),
            torch.randn(1, skip_list[2], h_in * 8, w_in * 8),
            torch.randn(1, skip_list[3], h_in * 16, w_in * 16),
        ]
        output: torch.Tensor = decoder(x, skips)
        assert output.shape[1] == decoder.out_channels
        assert output.shape[2:] == skips[-1].shape[2:]

    def test_forward_edge_case_skip_shapes_completes(self) -> None:
        in_ch: int = 128
        skip_list: list[int] = [64]  # Single skip
        h_in: int = 16
        w_in: int = 16
        decoder = CNNDecoder(in_ch, skip_list)
        x: torch.Tensor = torch.randn(3, in_ch, h_in, w_in)
        skips: list[torch.Tensor] = [
            torch.randn(3, skip_list[0], h_in * 2, w_in * 2)
        ]
        output: torch.Tensor = decoder(x, skips)
        assert output.shape[1] == decoder.out_channels
        assert output.shape[2:] == skips[-1].shape[2:]


def test_cnndecoder_different_input_sizes() -> None:
    """Test decoder with different input spatial sizes."""
    in_ch: int = 128
    skip_channels_list: list[int] = [32, 16, 8]  # Descending
    decoder = CNNDecoder(in_ch, skip_channels_list, out_channels=1)

    input_sizes: list[tuple[int, int]] = [(2, 2), (4, 4), (8, 8), (16, 16)]

    for h_in, w_in in input_sizes:
        x: torch.Tensor = torch.randn(1, in_ch, h_in, w_in)
        skips: list[torch.Tensor] = []

        # Create skips with appropriate sizes
        for i, ch in enumerate(skip_channels_list):
            skip_h: int = h_in * (2 ** (i + 1))
            skip_w: int = w_in * (2 ** (i + 1))
            skips.append(torch.randn(1, ch, skip_h, skip_w))

        output: torch.Tensor = decoder(x, skips)

        # Output should be 8x the input size (2^3 for 3 decoder blocks)
        expected_h: int = h_in * (2 ** len(skip_channels_list))
        expected_w: int = w_in * (2 ** len(skip_channels_list))
        assert output.shape == (1, 1, expected_h, expected_w)


def test_cnndecoder_segmentation_output_dimensions() -> None:
    """Test decoder output dimensions for segmentation tasks."""
    configs: list[tuple[int, list[int], int, tuple[int, int]]] = [
        # (in_ch, skip_channels, num_classes, input_size)
        (128, [64, 32], 2, (8, 8)),  # Binary segmentation
        (256, [128, 64, 32], 5, (4, 4)),  # Multi-class
        (512, [256, 128], 10, (16, 16)),  # Many classes
    ]

    for in_ch, skip_channels, num_classes, input_size in configs:
        decoder = CNNDecoder(in_ch, skip_channels, out_channels=num_classes)

        # Create input
        x: torch.Tensor = torch.randn(2, in_ch, *input_size)

        # Create skips
        skips: list[torch.Tensor] = []
        for i, ch in enumerate(skip_channels):
            skip_h: int = input_size[0] * (2 ** (i + 1))
            skip_w: int = input_size[1] * (2 ** (i + 1))
            skips.append(torch.randn(2, ch, skip_h, skip_w))

        # Forward pass
        output: torch.Tensor = decoder(x, skips)

        # Check output shape
        expected_h: int = input_size[0] * (2 ** len(skip_channels))
        expected_w: int = input_size[1] * (2 ** len(skip_channels))
        assert output.shape == (2, num_classes, expected_h, expected_w)

        # Verify output is suitable for loss computation
        assert output.dtype == torch.float32
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


def test_cnndecoder_end_to_end_flow() -> None:
    """Test complete end-to-end flow through decoder."""
    # Simulate bottleneck output
    bottleneck_ch: int = 256
    bottleneck_h: int = 8
    bottleneck_w: int = 8

    # Define architecture
    skip_channels: list[int] = [128, 64, 32, 16]  # Descending order
    num_classes: int = 3

    # Create decoder
    decoder = CNNDecoder(
        bottleneck_ch, skip_channels, out_channels=num_classes
    )

    # Create inputs
    batch_size: int = 4
    x: torch.Tensor = torch.randn(
        batch_size, bottleneck_ch, bottleneck_h, bottleneck_w
    )

    # Create skip connections (simulating encoder outputs)
    skips: list[torch.Tensor] = []
    for i, ch in enumerate(skip_channels):
        skip_h: int = bottleneck_h * (2 ** (i + 1))
        skip_w: int = bottleneck_w * (2 ** (i + 1))
        skips.append(torch.randn(batch_size, ch, skip_h, skip_w))

    # Forward pass
    output: torch.Tensor = decoder(x, skips)

    # Verify output
    final_h: int = bottleneck_h * (2 ** len(skip_channels))
    final_w: int = bottleneck_w * (2 ** len(skip_channels))
    assert output.shape == (batch_size, num_classes, final_h, final_w)

    # Test gradient flow
    loss: torch.Tensor = output.mean()
    loss.backward()

    # Check gradients exist
    param: torch.nn.Parameter
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
    in_ch_param: int,
    skip_channels_list_param: list[int],
    input_size_param: tuple[int, int],
    batch_size_param: int,
    out_ch_param: int,
) -> None:
    """Parametrized test for final output dimensions."""
    decoder = CNNDecoder(
        in_ch_param, skip_channels_list_param, out_channels=out_ch_param
    )

    # Create input
    x: torch.Tensor = torch.randn(
        batch_size_param, in_ch_param, *input_size_param
    )

    # Create skips
    skips: list[torch.Tensor] = []
    h: int = input_size_param[0] * 2
    w: int = input_size_param[1] * 2
    for i, ch in enumerate(skip_channels_list_param):
        skips.append(torch.randn(batch_size_param, ch, h * (2**i), w * (2**i)))

    # Forward pass
    output: torch.Tensor = decoder(x, skips)

    # Verify dimensions
    expected_h: int = input_size_param[0] * (
        2 ** len(skip_channels_list_param)
    )
    expected_w: int = input_size_param[1] * (
        2 ** len(skip_channels_list_param)
    )
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
    in_ch_param: int,
    skip_channels_list_param: list[int],
    input_size_param: tuple[int, int],
    batch_size_param: int,
) -> None:
    """Test information flow through the decoder pipeline."""
    decoder = CNNDecoder(in_ch_param, skip_channels_list_param)

    # Create input with known pattern
    x: torch.Tensor = (
        torch.ones(batch_size_param, in_ch_param, *input_size_param) * 0.5
    )

    # Create skips with different patterns
    skips: list[torch.Tensor] = []
    h: int = input_size_param[0] * 2
    w: int = input_size_param[1] * 2
    for i, ch in enumerate(skip_channels_list_param):
        # Each skip has a different value to track information flow
        skip_value: float = (i + 1) * 0.1
        skips.append(
            torch.ones(batch_size_param, ch, h * (2**i), w * (2**i))
            * skip_value
        )

    # Forward pass
    output: torch.Tensor = decoder(x, skips)

    # Output should be influenced by all inputs
    assert output.shape[0] == batch_size_param
    assert output.shape[1] == decoder.out_channels

    # Check that output has been transformed (not just passthrough)
    assert not torch.allclose(output, torch.ones_like(output) * 0.5)
    assert output.min() != output.max()  # Has variation
