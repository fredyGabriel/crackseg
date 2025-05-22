"""Utility functions for channel dimension calculation and validation in
decoder architectures.

References:
- Decoder analysis: outputs/prd_project_refinement/test_suite_evaluation/\
    reports/decoder_analysis/decoder_component_analysis.md
- Task 5 (DecoderBlock refactor), Task 6.1 (CNNDecoder analysis)
"""


def calculate_decoder_channels(
    in_channels: int,
    skip_channels_list: list[int],
    scaling_factors: list[float] | None = None,
) -> list[int]:
    """
    Calculate output channels for each decoder stage.
    Args:
        in_channels: Number of input channels to the first decoder block
            (bottleneck).
        skip_channels_list: List of skip connection channels
            (low to high resolution).
        scaling_factors: Optional list of scaling factors per stage
            (default: 0.5 for each).
    Returns:
        List of output channels for each decoder block.
    Raises:
        ValueError: If input values are invalid or inconsistent.
    """
    if not isinstance(in_channels, int) or in_channels <= 0:
        raise ValueError(
            "in_channels must be a positive integer, got " f"{in_channels}"
        )
    if not skip_channels_list or not all(
        isinstance(c, int) and c > 0 for c in skip_channels_list
    ):
        raise ValueError(
            "skip_channels_list must be a non-empty list of "
            "positive integers"
        )
    n_blocks = len(skip_channels_list)
    if scaling_factors is not None:
        if len(scaling_factors) != n_blocks:
            raise ValueError(
                "scaling_factors must match skip_channels_list " "length"
            )
    else:
        scaling_factors = [0.5] * n_blocks
    channels = [in_channels]
    for i in range(n_blocks):
        next_ch = int(channels[-1] * scaling_factors[i])
        if next_ch <= 0:
            raise ValueError(
                f"Calculated channel at block {i} is not "
                f"positive: {next_ch}"
            )
        channels.append(next_ch)
    return channels[1:]


def validate_channel_dimensions(
    expected: int, actual: int, context: str = ""
) -> None:
    """
    Validate that expected and actual channel dimensions match.
    Args:
        expected: Expected channel dimension.
        actual: Actual channel dimension.
        context: Optional context string for error messages.
    Raises:
        ValueError: If dimensions do not match.
    """
    if expected != actual:
        msg = f"Channel dimension mismatch: expected {expected}, got {actual}."
        if context:
            msg += f" Context: {context}"
        raise ValueError(msg)


def validate_skip_channels_order(skip_channels_list: list[int]) -> None:
    """
    Ensure skip_channels_list is in ascending order (low to high resolution).
    Args:
        skip_channels_list: List of skip connection channels.
    Raises:
        ValueError: If the list is not sorted in ascending order.
    """
    if any(
        skip_channels_list[i] > skip_channels_list[i + 1]
        for i in range(len(skip_channels_list) - 1)
    ):
        raise ValueError(
            "skip_channels_list must be in ascending order "
            "(low to high resolution)"
        )
