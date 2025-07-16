import pytest

from crackseg.model.decoder.common.channel_utils import (
    calculate_decoder_channels,
    validate_channel_dimensions,
    validate_skip_channels_order,
)


def test_calculate_decoder_channels_default():
    """Test default scaling (0.5) for 3 blocks."""
    result = calculate_decoder_channels(64, [32, 16, 8])
    assert result == [32, 16, 8]


def test_calculate_decoder_channels_custom_scaling():
    """Test custom scaling factors."""
    result = calculate_decoder_channels(
        100, [50, 25], scaling_factors=[0.6, 0.4]
    )
    assert result == [60, 24]


def test_calculate_decoder_channels_invalid_in_channels():
    with pytest.raises(
        ValueError, match="in_channels must be a positive integer"
    ):
        calculate_decoder_channels(0, [32, 16])


def test_calculate_decoder_channels_invalid_skip_channels():
    with pytest.raises(
        ValueError, match="skip_channels_list must be a non-empty list"
    ):
        calculate_decoder_channels(64, [])
    with pytest.raises(
        ValueError, match="skip_channels_list must be a non-empty list"
    ):
        calculate_decoder_channels(64, [32, -1])


def test_calculate_decoder_channels_scaling_length_mismatch():
    with pytest.raises(
        ValueError,
        match="scaling_factors must match skip_channels_list length",
    ):
        calculate_decoder_channels(64, [32, 16], scaling_factors=[0.5])


def test_calculate_decoder_channels_negative_result():
    with pytest.raises(
        ValueError, match="Calculated channel at block 0 is not positive"
    ):
        calculate_decoder_channels(2, [1], scaling_factors=[0.0])


def test_validate_channel_dimensions_match():
    validate_channel_dimensions(16, 16)


def test_validate_channel_dimensions_mismatch():
    with pytest.raises(
        ValueError, match="Channel dimension mismatch: expected 16, got 8."
    ):
        validate_channel_dimensions(16, 8)
    with pytest.raises(ValueError, match="Context: DecoderBlock 2"):
        validate_channel_dimensions(16, 8, context="DecoderBlock 2")


def test_validate_skip_channels_order_valid():
    validate_skip_channels_order([32, 16, 8])
    validate_skip_channels_order([512, 256, 128, 64])
    validate_skip_channels_order([1])


def test_validate_skip_channels_order_invalid():
    with pytest.raises(
        ValueError, match="skip_channels_list must be in descending order"
    ):
        validate_skip_channels_order([8, 16, 32])
    with pytest.raises(
        ValueError, match="skip_channels_list must be in descending order"
    ):
        validate_skip_channels_order([16, 32, 8])
