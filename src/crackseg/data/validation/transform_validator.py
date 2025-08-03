"""Transform configuration validation utilities.

This module provides comprehensive validation functions for transform
configuration parameters used throughout the crack segmentation pipeline. It
ensures configuration integrity, prevents runtime errors, and provides clear
feedback for configuration issues.

Key Features:
    - Transform configuration validation with essential transform checking
    - Parameter validation for specific transforms (Resize, Normalize)
    - Format conversion utilities for Albumentations compatibility
    - Flexible input format support (dict, DictConfig, list, ListConfig)

Core Functions:
    - validate_transform_config(): Validates transformation pipeline
      configuration

Validation Scope:
    - Essential transform presence verification (Resize, Normalize)
    - Transform parameter completeness and format
    - Configuration format normalization
    - Parameter type and value validation

Common Usage:
    # Transform configuration validation
    validate_transform_config(cfg.transforms.train, mode="train")
    validate_transform_config(cfg.transforms.val, mode="val")

    # Works with various configuration formats
    config_dict = {"resize": {"height": 512, "width": 512}}
    validate_transform_config(config_dict)

Integration:
    - Used by factory functions before transform creation
    - Integrated with Hydra configuration system
    - Compatible with OmegaConf DictConfig and ListConfig
    - Provides early validation to prevent pipeline failures

Error Handling:
    - Clear, descriptive error messages for all validation failures
    - Specific parameter identification in error contexts
    - Warnings for non-critical configuration issues
    - Type-specific validation with appropriate error types

References:
    - Factory: src.data.factory.create_dataloaders_from_config
    - Transforms: src.data.transforms.get_transforms_from_config
    - Configuration: configs/transforms/
"""

import warnings
from typing import Any

from omegaconf import DictConfig, ListConfig

from .format_converter import _convert_albumentations_format_to_standard
from .parameter_validators import (
    _validate_normalize_params,
    _validate_resize_params,
)


def validate_transform_config(
    transform_config: (
        list[dict[str, Any] | DictConfig]
        | ListConfig
        | dict[str, Any]
        | DictConfig
    ),
    mode_unused: (
        str | None
    ) = None,  # Mode parameter is not used in the current logic
) -> None:
    """Validate comprehensive transform configuration for pipeline reliability.

    Performs thorough validation of transformation pipeline configuration,
    ensuring essential transforms are present with valid parameters. Supports
    multiple configuration formats and provides detailed error reporting.

    Args:
        transform_config: Transform configuration in flexible formats:
            - list[dict]: List of transform specifications with 'name' and
              'params'
            - ListConfig: Hydra list configuration
            - dict: Dictionary mapping transform names to parameters
            - DictConfig: Hydra dictionary configuration
        mode_unused: Optional mode specification ('train', 'val', 'test').
            Reserved for future mode-specific validation logic.

    Raises:
        ValueError: If configuration is invalid or essential transforms are
            missing:
            - Missing required 'Resize' transform when transforms are
              specified
            - Missing required 'Normalize' transform when transforms are
              specified
            - Invalid parameters for specific transforms (delegated to
              specific validators)

    Warnings:
        UserWarning: If transform configuration list is empty, which may
            cause issues if Resize and Normalize transforms are mandatory
            for the pipeline.

    Examples:
        >>> # Validate standard list format
        >>> config = [
        ...     {"name": "Resize", "params": {"height": 512, "width": 512}},
        ...     {"name": "Normalize", "params": {"mean": [0.485, 0.456, 0.406],
        ...                                       "std": [0.229, 0.224, 0.225]}}
        ... ]
        >>> validate_transform_config(config)
        >>> # No errors raised

        >>> # Validate dictionary format
        >>> config = {
        ...     "Resize": {"height": 512, "width": 512},
        ...     "Normalize": {"mean": [0.485, 0.456, 0.406],
        ...                   "std": [0.229, 0.224, 0.225]}
        ... }
        >>> validate_transform_config(config)
        >>> # No errors raised

        >>> # Validate with missing essential transforms
        >>> config = {"RandomCrop": {"height": 256, "width": 256}}
        >>> validate_transform_config(config)
        >>> # ValueError: Missing required 'Resize' transform

    Note:
        This function automatically handles format conversion and parameter
        validation. It supports various input formats and provides clear
        error messages for configuration issues.

        Essential transforms (Resize, Normalize) are required when any
        transforms are specified. This ensures consistent data preprocessing
        across the pipeline.

        Parameter validation is delegated to specific validator functions
        for each transform type, providing detailed validation for complex
        parameters.
    """
    # Convert to standard format for consistent processing
    try:
        transforms_list = _convert_albumentations_format_to_standard(
            transform_config
        )
    except Exception as e:
        raise ValueError(
            f"Failed to convert transform configuration: {e}"
        ) from e

    # Check for empty configuration
    if not transforms_list:
        warnings.warn(
            "Transform configuration is empty. This may cause issues if "
            "Resize and Normalize transforms are mandatory for the pipeline.",
            stacklevel=2,
        )
        return

    # Track found transforms
    found_transforms = set()
    transform_params = {}

    # Process each transform
    for transform_spec in transforms_list:
        if isinstance(transform_spec, dict):
            name = transform_spec.get("name")
            params = transform_spec.get("params", {})
        elif isinstance(transform_spec, DictConfig):
            # Handle DictConfig objects
            spec_dict = dict(transform_spec)
            name = spec_dict.get("name")
            params = spec_dict.get("params", {})
        else:
            raise ValueError(
                f"Invalid transform specification format: {type(transform_spec)}"
            )

        if not name:
            raise ValueError("Transform specification must contain 'name'")

        found_transforms.add(name.lower())
        transform_params[name] = params

        # Validate specific transform parameters
        if name.lower() == "resize":
            _validate_resize_params(params)
        elif name.lower() == "normalize":
            _validate_normalize_params(params)

    # Check for essential transforms
    if found_transforms and "resize" not in found_transforms:
        raise ValueError(
            "Missing required 'Resize' transform. "
            "Resize transform is mandatory for consistent image dimensions."
        )

    if found_transforms and "normalize" not in found_transforms:
        raise ValueError(
            "Missing required 'Normalize' transform. "
            "Normalize transform is mandatory for proper model input scaling."
        )
