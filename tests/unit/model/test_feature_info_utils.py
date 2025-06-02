"""Tests for feature info utilities."""

from src.model.encoder.feature_info_utils import (
    build_feature_info_from_channels,
    create_feature_info_entry,
    validate_feature_info,
)


class TestFeatureInfoUtils:
    """Test suite for feature info utility functions."""

    def test_create_feature_info_entry_basic(self) -> None:
        """Test basic feature info entry creation."""
        entry = create_feature_info_entry(channels=64, reduction=4, stage=0)

        expected = {
            "channels": 64,
            "reduction": 4,
            "stage": 0,
        }
        assert entry == expected

    def test_create_feature_info_entry_with_name(self) -> None:
        """Test feature info entry creation with name."""
        entry = create_feature_info_entry(
            channels=128, reduction=8, stage=1, name="stage_1"
        )

        expected = {
            "channels": 128,
            "reduction": 8,
            "stage": 1,
            "name": "stage_1",
        }
        assert entry == expected

    def test_build_feature_info_from_channels_basic(self) -> None:
        """Test building feature info from channel lists."""
        skip_channels = [64, 128, 256]
        out_channels = 512

        feature_info = build_feature_info_from_channels(
            skip_channels=skip_channels,
            out_channels=out_channels,
        )

        expected = [
            {"channels": 64, "reduction": 4, "stage": 0},
            {"channels": 128, "reduction": 8, "stage": 1},
            {"channels": 256, "reduction": 16, "stage": 2},
            {
                "channels": 512,
                "reduction": 32,
                "stage": 3,
                "name": "bottleneck",
            },
        ]
        assert feature_info == expected

    def test_build_feature_info_custom_reduction(self) -> None:
        """Test building feature info with custom reduction factors."""
        skip_channels = [32, 64]
        out_channels = 128

        feature_info = build_feature_info_from_channels(
            skip_channels=skip_channels,
            out_channels=out_channels,
            base_reduction=2,
            reduction_factor=2,
        )

        expected = [
            {"channels": 32, "reduction": 2, "stage": 0},
            {"channels": 64, "reduction": 4, "stage": 1},
            {
                "channels": 128,
                "reduction": 8,
                "stage": 2,
                "name": "bottleneck",
            },
        ]
        assert feature_info == expected

    def test_build_feature_info_with_stage_names(self) -> None:
        """Test building feature info with custom stage names."""
        skip_channels = [64, 128]
        out_channels = 256
        stage_names = ["encoder_0", "encoder_1", "final"]

        feature_info = build_feature_info_from_channels(
            skip_channels=skip_channels,
            out_channels=out_channels,
            stage_names=stage_names,
        )

        expected = [
            {"channels": 64, "reduction": 4, "stage": 0, "name": "encoder_0"},
            {"channels": 128, "reduction": 8, "stage": 1, "name": "encoder_1"},
            {"channels": 256, "reduction": 16, "stage": 2, "name": "final"},
        ]
        assert feature_info == expected

    def test_validate_feature_info_valid(self) -> None:
        """Test validation of valid feature info."""
        valid_info = [
            {"channels": 64, "reduction": 4, "stage": 0},
            {"channels": 128, "reduction": 8, "stage": 1, "name": "stage_1"},
        ]

        assert validate_feature_info(valid_info) is True

    def test_validate_feature_info_missing_required_field(self) -> None:
        """Test validation fails with missing required fields."""
        invalid_info = [
            {"channels": 64, "reduction": 4},  # Missing "stage"
        ]

        assert validate_feature_info(invalid_info) is False

    def test_validate_feature_info_invalid_channel_type(self) -> None:
        """Test validation fails with invalid channel type."""
        invalid_info = [
            {
                "channels": "64",
                "reduction": 4,
                "stage": 0,
            },  # String instead of int
        ]

        assert validate_feature_info(invalid_info) is False

    def test_validate_feature_info_negative_values(self) -> None:
        """Test validation fails with negative values."""
        invalid_info = [
            {"channels": -64, "reduction": 4, "stage": 0},  # Negative channels
        ]

        assert validate_feature_info(invalid_info) is False

    def test_validate_feature_info_wrong_stage_order(self) -> None:
        """Test validation fails with wrong stage ordering."""
        invalid_info = [
            {"channels": 64, "reduction": 4, "stage": 1},  # Should be stage 0
        ]

        assert validate_feature_info(invalid_info) is False

    def test_validate_feature_info_invalid_name_type(self) -> None:
        """Test validation fails with invalid name type."""
        invalid_info = [
            {
                "channels": 64,
                "reduction": 4,
                "stage": 0,
                "name": 123,
            },  # Number instead of string
        ]

        assert validate_feature_info(invalid_info) is False

    def test_build_and_validate_integration(self) -> None:
        """Test integration between build and validate functions."""
        skip_channels = [32, 64, 128]
        out_channels = 256

        feature_info = build_feature_info_from_channels(
            skip_channels=skip_channels,
            out_channels=out_channels,
        )

        # Built feature info should always be valid
        assert validate_feature_info(feature_info) is True

        # Check expected structure
        assert len(feature_info) == 4  # 3 skip + 1 bottleneck
        assert all("channels" in info for info in feature_info)
        assert all("reduction" in info for info in feature_info)
        assert all("stage" in info for info in feature_info)
