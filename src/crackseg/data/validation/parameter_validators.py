"""Parameter validation utilities for transform configuration.

This module provides specific parameter validation functions for individual
transforms used in the crack segmentation pipeline.
"""

import warnings
from typing import Any

from omegaconf import DictConfig


def _validate_resize_params(params: dict[str, Any] | DictConfig) -> None:
    """Validate Resize transform parameters.

    Args:
        params: Resize transform parameters.

    Raises:
        ValueError: If parameters are invalid.
    """
    if not isinstance(params, dict | DictConfig):
        raise ValueError("Resize parameters must be a dictionary")

    # Convert to dict if needed
    if isinstance(params, DictConfig):
        params = dict(params)  # type: ignore[assignment]

    # Validate required parameters
    if "height" not in params:
        raise ValueError("Resize transform requires 'height' parameter")
    if "width" not in params:
        raise ValueError("Resize transform requires 'width' parameter")

    # Validate parameter types and values
    height = params["height"]
    width = params["width"]

    if not isinstance(height, int) or height <= 0:
        raise ValueError(
            f"Resize height must be positive integer, got {height}"
        )
    if not isinstance(width, int) or width <= 0:
        raise ValueError(f"Resize width must be positive integer, got {width}")

    # Validate reasonable size limits
    if height > 4096 or width > 4096:
        warnings.warn(
            f"Large resize dimensions detected: {width}x{height}. "
            "This may cause memory issues.",
            stacklevel=2,
        )

    if height < 32 or width < 32:
        warnings.warn(
            f"Small resize dimensions detected: {width}x{height}. "
            "This may affect model performance.",
            stacklevel=2,
        )


def _validate_normalize_params(params: dict[str, Any] | DictConfig) -> None:
    """Validate Normalize transform parameters.

    Args:
        params: Normalize transform parameters.

    Raises:
        ValueError: If parameters are invalid.
    """
    if not isinstance(params, dict | DictConfig):
        raise ValueError("Normalize parameters must be a dictionary")

    # Convert to dict if needed
    if isinstance(params, DictConfig):
        params = dict(params)  # type: ignore[assignment]

    # Validate required parameters
    if "mean" not in params:
        raise ValueError("Normalize transform requires 'mean' parameter")
    if "std" not in params:
        raise ValueError("Normalize transform requires 'std' parameter")

    # Validate parameter types and values
    mean = params["mean"]
    std = params["std"]

    # Validate mean parameter
    if isinstance(mean, list | tuple):
        if len(mean) != 3:
            raise ValueError(
                f"Normalize mean must have 3 values for RGB, got {len(mean)}"
            )
        for val in mean:
            if not isinstance(val, int | float):
                raise ValueError(
                    f"Normalize mean values must be numeric, got {val}"
                )
    elif isinstance(mean, int | float):
        # Single value - convert to list
        mean = [float(mean)] * 3
    else:
        raise ValueError(
            f"Normalize mean must be numeric or list, got {type(mean)}"
        )

    # Validate std parameter
    if isinstance(std, list | tuple):
        if len(std) != 3:
            raise ValueError(
                f"Normalize std must have 3 values for RGB, got {len(std)}"
            )
        for val in std:
            if not isinstance(val, int | float) or val <= 0:
                raise ValueError(
                    f"Normalize std values must be positive, got {val}"
                )
    elif isinstance(std, int | float):
        if std <= 0:
            raise ValueError(f"Normalize std must be positive, got {std}")
        # Single value - convert to list
        std = [float(std)] * 3
    else:
        raise ValueError(
            f"Normalize std must be numeric or list, got {type(std)}"
        )

    # Validate normalization ranges
    for i, (m, s) in enumerate(zip(mean, std, strict=False)):
        if abs(m) > 255:
            warnings.warn(
                f"Normalize mean[{i}]={m} is outside typical range [0, 255]",
                stacklevel=2,
            )
        if s > 255:
            warnings.warn(
                f"Normalize std[{i}]={s} is outside typical range [0, 255]",
                stacklevel=2,
            )
