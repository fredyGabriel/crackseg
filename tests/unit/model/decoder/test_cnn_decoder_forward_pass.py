import pytest
import torch

from src.model.decoder.cnn_decoder import CNNDecoder


def test_cnndecoder_forward_shape():
    """Test forward pass output shape."""
    batch_size = 2
    bottleneck_channels = 64
    skip_channels_list = [8, 16, 32]  # Ascending
    decoder = CNNDecoder(bottleneck_channels, skip_channels_list)
    h_bottleneck, w_bottleneck = 8, 8
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


def test_cnndecoder_output_shape_various_configs():
    """Test output shape for various channel and depth configurations."""
    configs = [
        # (in_ch, skip_channels_list (ASC), out_channels_final)
        (16, [4, 8], 1),
        (32, [4, 8, 16], 2),
        (64, [4, 8, 16, 32], 3),
    ]
    for in_ch, skip_channels_list_config, out_channels_final_config in configs:
        decoder = CNNDecoder(
            in_ch,
            skip_channels_list_config,
            out_channels=out_channels_final_config,
        )
        assert len(decoder.decoder_blocks) == len(skip_channels_list_config)

        x = torch.randn(1, in_ch, 8, 8)
        skip_tensors = []
        h_skip_base, w_skip_base = 8 * 2, 8 * 2  # Double the initial input
        for i, skip_ch_val in enumerate(skip_channels_list_config):
            skip_tensors.append(
                torch.randn(
                    1, skip_ch_val, h_skip_base * (2**i), w_skip_base * (2**i)
                )
            )

        output = decoder(x, skip_tensors)
        expected_h, expected_w = skip_tensors[-1].shape[2:]
        assert output.shape[0] == 1
        assert output.shape[1] == out_channels_final_config
        assert output.shape[2] == expected_h
        assert output.shape[3] == expected_w


class CNNDecoderForwardTests:
    """Tests adapted for ascending skips and increasing dimensions."""

    def test_forward_standard_skip_shape_completes(self):
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

    def test_forward_varied_skip_shapes_completes(self):
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

    def test_forward_edge_case_skip_shapes_completes(self):
        in_ch = 128
        skip_list = [64]  # Single skip
        h_in, w_in = 16, 16
        decoder = CNNDecoder(in_ch, skip_list)
        x = torch.randn(3, in_ch, h_in, w_in)
        skips = [torch.randn(3, skip_list[0], h_in * 2, w_in * 2)]
        output = decoder(x, skips)
        assert output.shape[1] == decoder.out_channels
        assert output.shape[2:] == skips[-1].shape[2:]


def test_cnndecoder_different_input_sizes():
    """Test CNNDecoder with different input spatial sizes (ascending skips)."""
    in_ch = 64
    skip_channels_list = [8, 16, 32]  # Ascending
    decoder = CNNDecoder(in_ch, skip_channels_list, out_channels=1)
    input_spatial_sizes = [(8, 8), (16, 16), (7, 9), (32, 24)]

    for h_in_loop, w_in_loop in input_spatial_sizes:
        x = torch.randn(1, in_ch, h_in_loop, w_in_loop)
        skip_tensors = []
        current_h, current_w = h_in_loop * 2, w_in_loop * 2
        for i, skip_ch_val in enumerate(skip_channels_list):
            skip_tensors.append(
                torch.randn(
                    1, skip_ch_val, current_h * (2**i), current_w * (2**i)
                )
            )
        output = decoder(x, skip_tensors)
        expected_h, expected_w = skip_tensors[-1].shape[2:]
        assert output.shape[2:] == (expected_h, expected_w)


def test_cnndecoder_segmentation_output_dimensions():
    """Test CNNDecoder output for a typical segmentation task."""
    batch_size = 4
    bottleneck_depth_channels = 512  # Bottleneck channels
    skip_config_channels = [64, 128, 256]  # Encoder skips (ascending)
    num_classes = 5  # Number of classes for segmentation

    decoder = CNNDecoder(
        bottleneck_depth_channels,
        skip_config_channels,
        out_channels=num_classes,
    )

    h_bottleneck_seg, w_bottleneck_seg = 32, 32
    bottleneck_features = torch.randn(
        batch_size,
        bottleneck_depth_channels,
        h_bottleneck_seg,
        w_bottleneck_seg,
    )

    skip_feature_list = []
    current_h_skip, current_w_skip = h_bottleneck_seg * 2, w_bottleneck_seg * 2
    for i, sc_channels in enumerate(skip_config_channels):
        skip_feature_list.append(
            torch.randn(
                batch_size,
                sc_channels,
                current_h_skip * (2**i),
                current_w_skip * (2**i),
            )
        )

    segmentation_output = decoder(bottleneck_features, skip_feature_list)

    final_h, final_w = skip_feature_list[-1].shape[2:]
    assert segmentation_output.shape == (
        batch_size,
        num_classes,
        final_h,
        final_w,
    )


def test_cnndecoder_end_to_end_flow():
    """Test end-to-end flow with a typical U-Net like configuration."""
    batch_size = 2
    bottleneck_channels = 256
    encoder_skip_channels = [32, 64, 128]  # Ascending for CNNDecoder
    skip_channels_list_for_decoder = encoder_skip_channels

    final_out_channels = 3
    decoder = CNNDecoder(
        bottleneck_channels,
        skip_channels_list_for_decoder,
        out_channels=final_out_channels,
    )

    h_bottleneck_e2e, w_bottleneck_e2e = 16, 16
    bottleneck_tensor = torch.randn(
        batch_size, bottleneck_channels, h_bottleneck_e2e, w_bottleneck_e2e
    )

    skip_tensors_list = []
    current_h_e2e, current_w_e2e = h_bottleneck_e2e * 2, w_bottleneck_e2e * 2
    for i, skip_ch_val in enumerate(skip_channels_list_for_decoder):
        skip_tensors_list.append(
            torch.randn(
                batch_size,
                skip_ch_val,
                current_h_e2e * (2**i),
                current_w_e2e * (2**i),
            )
        )

    output_segmentation = decoder(bottleneck_tensor, skip_tensors_list)

    expected_h, expected_w = skip_tensors_list[-1].shape[2:]
    assert output_segmentation.shape == (
        batch_size,
        final_out_channels,
        expected_h,
        expected_w,
    )
    assert not torch.isnan(output_segmentation).any()
    assert not torch.isinf(output_segmentation).any()


@pytest.mark.parametrize(
    "in_ch_param, skip_channels_list_param, input_size_param, "
    "batch_size_param, out_channels_param",
    [
        (32, [8, 16], (8, 8), 2, 1),
        (64, [8, 16, 32], (16, 16), 1, 3),
        (16, [4, 8], (7, 9), 3, 2),
        (128, [8, 16, 32, 64], (4, 4), 1, 4),
    ],
)
def test_final_output_dimensions_parametrized(
    in_ch_param,
    skip_channels_list_param,
    input_size_param,
    batch_size_param,
    out_channels_param,
):
    decoder = CNNDecoder(
        in_ch_param, skip_channels_list_param, out_channels=out_channels_param
    )
    h_in, w_in = input_size_param
    x = torch.randn(batch_size_param, in_ch_param, h_in, w_in)
    skips = []
    current_h, current_w = h_in * 2, w_in * 2
    for i, skip_ch in enumerate(skip_channels_list_param):
        skips.append(
            torch.randn(
                batch_size_param,
                skip_ch,
                current_h * (2**i),
                current_w * (2**i),
            )
        )
    output = decoder(x, skips)
    expected_h, expected_w = skips[-1].shape[2:]
    assert output.shape == (
        batch_size_param,
        out_channels_param,
        expected_h,
        expected_w,
    )


@pytest.mark.parametrize(
    "in_ch_param, skip_channels_list_param, input_size_param, "
    "batch_size_param",
    [
        (32, [8, 16], (8, 8), 2),
        (64, [8, 16, 32], (16, 16), 1),
        (16, [4, 8], (7, 9), 3),
        (128, [8, 16, 32, 64], (4, 4), 1),
    ],
)
def test_information_flow_pipeline_parametrized(
    in_ch_param, skip_channels_list_param, input_size_param, batch_size_param
):
    # Default out_channels
    decoder = CNNDecoder(in_ch_param, skip_channels_list_param)
    h_in, w_in = input_size_param
    x = torch.randn(batch_size_param, in_ch_param, h_in, w_in)
    skips = []
    current_h, current_w = h_in * 2, w_in * 2
    for i, skip_ch in enumerate(skip_channels_list_param):
        skips.append(
            torch.randn(
                batch_size_param,
                skip_ch,
                current_h * (2**i),
                current_w * (2**i),
            )
        )
    output = decoder(x, skips)
    # Check output is not all zeros (basic information flow check)
    assert not torch.all(output == 0)
    # Check correct number of channels (default out_channels = 1)
    # or decoder.out_channels if it's reliably 1 by default
    assert output.size(1) == 1
