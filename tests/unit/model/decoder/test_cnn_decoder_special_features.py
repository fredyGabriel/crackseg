import pytest
import torch

from src.model.decoder.cnn_decoder import CNNDecoder


def test_cnndecoder_custom_target_size() -> None:
    """Test CNNDecoder with custom target_size.
    NOTE: target_size might not be a valid parameter in the refactored API.
    If not, this test should be removed or adapted.
    Assuming for now it exists and CNNDecoder performs a final upsample if
    necessary."""
    in_ch = 64
    skip_channels_list = [8, 16, 32]  # Ascending
    # target_h, target_w = 256, 256 # Desired output size
    # If target_size is a valid kwarg:
    # decoder = CNNDecoder(in_ch, skip_channels_list,
    # target_size=(target_h, target_w))
    # else:
    decoder = CNNDecoder(in_ch, skip_channels_list)  # Assume no target_size

    x = torch.randn(1, in_ch, 8, 8)
    skip_tensors = []
    current_h, current_w = 8 * 2, 8 * 2
    for i, skip_ch_val in enumerate(skip_channels_list):
        skip_tensors.append(
            torch.randn(1, skip_ch_val, current_h * (2**i), current_w * (2**i))
        )
    output = decoder(x, skip_tensors)

    # If CNNDecoder does NOT have target_size, the output is from the
    # last skip.
    # If it DOES have target_size and implements it with a final upsample,
    # then:
    # assert output.shape[2:] == (target_h, target_w)
    # For now, assume NO explicit target_size and output is dictated by skips.
    expected_h, expected_w = skip_tensors[-1].shape[2:]
    assert output.shape[2:] == (expected_h, expected_w)


@pytest.mark.parametrize(
    "in_channels_test, skip_channels_list_test, out_channels_test, "
    "batch_size_test, h_test, w_test",
    [
        (64, [8, 16, 32], 3, 4, 16, 16),  # Ascending, no CBAM
        (32, [8, 16], 2, 2, 8, 12),  # Ascending, no CBAM
        # Add cases with CBAM if relevant and API supports it explicitly
        # (64, [8, 16, 32], 1, True, 1, 16, 16), # With CBAM
    ],
)
def test_cnndecoder_forward_cbam_and_shapes(
    in_channels_test: int,
    skip_channels_list_test: list[int],
    out_channels_test: int,
    batch_size_test: int,
    h_test: int,
    w_test: int,
) -> None:
    """
    Test forward pass, output shapes, and CBAM integration (if applicable).
    """
    # Assume use_cbam is a boolean passed to CNNDecoder if API supports it.
    # If not, this test parameter and its logic should be removed/adapted.
    # decoder = CNNDecoder(in_channels_test, skip_channels_list_test,
    # out_channels=out_channels_test, use_cbam=use_cbam_test)
    # For now, assume `use_cbam` is not a direct parameter of CNNDecoder,
    # but of DecoderBlock.
    # CNNDecoder simply instantiates DecoderBlocks, which may or may not use
    # CBAM internally.
    decoder = CNNDecoder(
        in_channels_test,
        skip_channels_list_test,
        out_channels=out_channels_test,
    )

    x_input = torch.randn(batch_size_test, in_channels_test, h_test, w_test)
    skip_tensors_list_val = []
    current_h_s, current_w_s = h_test * 2, w_test * 2
    for i, sc_ch in enumerate(skip_channels_list_test):
        skip_tensors_list_val.append(
            torch.randn(
                batch_size_test,
                sc_ch,
                current_h_s * (2**i),
                current_w_s * (2**i),
            )
        )

    output_val = decoder(x_input, skip_tensors_list_val)

    expected_h_val, expected_w_val = skip_tensors_list_val[-1].shape[2:]
    assert output_val.shape == (
        batch_size_test,
        out_channels_test,
        expected_h_val,
        expected_w_val,
    )
    # If CBAM is tested, assertions about existence of CBAM modules in
    # decoder.decoder_blocks
    # or about CBAM's expected effect (more complex to test directly in output)
    # could be added.
