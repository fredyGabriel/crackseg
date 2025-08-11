"""Utility helpers for ConvLSTM components.

These functions are extracted from the core ConvLSTM implementation to
improve modularity and reduce file size while preserving the public API.
"""

from __future__ import annotations

from typing import Any


def check_kernel_size_consistency(kernel_size: Any, expected_dims: int) -> Any:
    """Validate and normalize the ``kernel_size`` field.

    Accepts one of:
    - tuple of expected length containing ints
    - list/sequence of expected length (elements cast to int)
    - list of sequences of expected length (per-layer kernel sizes)

    Returns the input if valid (possibly normalized), otherwise raises ValueError.
    """

    # Case 1: Already a tuple of the right length consisting of ints
    if isinstance(kernel_size, tuple) and len(kernel_size) == expected_dims:
        if all(isinstance(elem, int) for elem in kernel_size):
            return kernel_size

    # Case 2: Sequence of numeric values -> convert to tuple of ints
    if hasattr(kernel_size, "__len__") and len(kernel_size) == expected_dims:
        try:
            as_tuple = tuple(int(elem) for elem in kernel_size)
            return as_tuple
        except (ValueError, TypeError):
            pass

    # Case 3: List of sequences (per-layer kernel sizes)
    is_list_of_valid_pairs = (
        hasattr(kernel_size, "__iter__")
        and hasattr(kernel_size, "__len__")
        and all(
            hasattr(elem, "__len__") and len(elem) == expected_dims
            for elem in kernel_size
        )
    )
    if is_list_of_valid_pairs:
        return kernel_size

    raise ValueError(
        f"kernel_size must be a tuple of {expected_dims} ints, "
        f"list of {expected_dims} ints, or list of tuples/lists for multi-layer ConvLSTM"
    )


def extend_param_for_layers(param: Any, num_layers: int) -> list[Any]:
    """Extend a non-list parameter or validate list length for ``num_layers``.

    - If ``param`` is not a list, returns a list repeating ``param`` ``num_layers`` times.
    - If it is a list, validates that its length matches ``num_layers``.
    """

    if not isinstance(param, list):
        return [param] * num_layers

    if len(param) != num_layers:
        raise ValueError(
            f"Length of list param ({len(param)}) doesn't match num_layers ({num_layers})"
        )
    return param
