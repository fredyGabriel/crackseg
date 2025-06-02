"""Feature info utilities for encoder components."""

from typing import Any


def create_feature_info_entry(
    channels: int,
    reduction: int,
    stage: int,
    name: str | None = None,
) -> dict[str, Any]:
    """
    Create a standardized feature info entry for encoders.

    Args:
        channels: Number of output channels for this stage
        reduction: Spatial reduction factor from input
        stage: Stage index
        name: Optional stage name

    Returns:
        Standardized feature info dictionary
    """
    entry: dict[str, Any] = {
        "channels": channels,
        "reduction": reduction,
        "stage": stage,
    }
    if name is not None:
        entry["name"] = name
    return entry


def build_feature_info_from_channels(
    skip_channels: list[int],
    out_channels: int,
    base_reduction: int = 4,
    reduction_factor: int = 2,
    stage_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Build standardized feature info from channel lists.

    Args:
        skip_channels: List of skip connection channels (high to low res)
        out_channels: Final output channels (bottleneck)
        base_reduction: Base reduction factor for first stage
        reduction_factor: Multiplication factor between stages
        stage_names: Optional list of stage names

    Returns:
        List of standardized feature info dictionaries
    """
    feature_info = []

    # Add info for each skip connection stage
    for i, channels in enumerate(skip_channels):
        reduction = base_reduction * (reduction_factor**i)
        name = stage_names[i] if stage_names and i < len(stage_names) else None
        feature_info.append(
            create_feature_info_entry(channels, reduction, i, name)
        )

    # Add info for the final output (bottleneck)
    final_stage = len(skip_channels)
    final_reduction = base_reduction * (reduction_factor**final_stage)
    final_name = (
        stage_names[final_stage]
        if stage_names and final_stage < len(stage_names)
        else "bottleneck"
    )
    feature_info.append(
        create_feature_info_entry(
            out_channels, final_reduction, final_stage, final_name
        )
    )

    return feature_info


def validate_feature_info(feature_info: list[dict[str, Any]]) -> bool:
    """
    Validate that feature info follows the expected structure.

    Args:
        feature_info: List of feature info dictionaries to validate

    Returns:
        True if valid, False otherwise
    """
    required_fields = {"channels", "reduction", "stage"}

    for i, info in enumerate(feature_info):
        # Check required fields
        if not required_fields.issubset(info.keys()):
            return False

        # Validate field types
        if not isinstance(info["channels"], int) or info["channels"] <= 0:
            return False

        if not isinstance(info["reduction"], int) or info["reduction"] <= 0:
            return False

        if not isinstance(info["stage"], int) or info["stage"] < 0:
            return False

        # Stage should match index
        if info["stage"] != i:
            return False

        # Name field is optional but should be string if present
        if "name" in info and not isinstance(info["name"], str):
            return False

    return True
