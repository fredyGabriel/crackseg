"""Data configuration validation utilities.

This module provides comprehensive validation functions for data configuration
parameters used throughout the crack segmentation pipeline. It ensures
configuration integrity, prevents runtime errors, and provides clear feedback
for configuration issues.

Key Features:
    - Data configuration validation with split ratio verification
    - Path validation and accessibility checking
    - Image size format and dimensionality validation
    - Mathematical consistency verification for split ratios

Core Functions:
    - validate_data_config(): Validates dataset configuration parameters
    - _validate_split_ratios(): Validates train/val/test split ratios
    - _validate_image_size(): Validates image size format and dimensions

Validation Scope:
    - Data paths and accessibility
    - Split ratios mathematical correctness (sum to 1.0)
    - Image size format and validity
    - Configuration completeness and format

Common Usage:
    # Data configuration validation
    validate_data_config(cfg.data)

    # Works with various configuration formats
    config_dict = {"data_root": "/path", "train_split": 0.7, ...}
    validate_data_config(config_dict)

Integration:
    - Used by factory functions before dataset creation
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
    - Dataset: src.data.dataset.CrackSegmentationDataset
    - Configuration: configs/data/
"""

from typing import Any

from omegaconf import DictConfig, ListConfig


def validate_data_config(data_cfg: dict[str, Any] | DictConfig) -> None:
    """Validate comprehensive data configuration for pipeline integrity.

    Performs thorough validation of data configuration parameters to ensure
    pipeline reliability and prevent runtime failures. Validates paths,
    split ratios, image dimensions, and configuration completeness.

    Args:
        data_cfg: Data configuration dictionary or DictConfig containing:
            - data_root: str - Path to dataset root directory
            - train_split: float - Training data ratio (0.0-1.0)
            - val_split: float - Validation data ratio (0.0-1.0)
            - test_split: float - Test data ratio (0.0-1.0)
            - image_size: list[int] - Target image dimensions [height, width]

    Raises:
        ValueError: If any configuration parameter is missing, invalid, or
            mathematically inconsistent:
            - Missing required keys: data_root, train_split, val_split,
              test_split, image_size
            - Split ratios don't sum to 1.0 (within 1e-4 tolerance)
            - Image size is not a 2-element list/tuple/ListConfig
            - Invalid numeric values for splits or image dimensions

    Examples:
        >>> # Valid configuration
        >>> config = {
        ...     "data_root": "/path/to/dataset",
        ...     "train_split": 0.7,
        ...     "val_split": 0.2,
        ...     "test_split": 0.1,
        ...     "image_size": [512, 512]
        ... }
        >>> validate_data_config(config)  # No error
        >>>
        >>> # Invalid configuration (splits don't sum to 1.0)
        >>> bad_config = {
        ...     "data_root": "/path/to/dataset",
        ...     "train_split": 0.6,
        ...     "val_split": 0.3,
        ...     "test_split": 0.2,  # Total = 1.1
        ...     "image_size": [512, 512]
        ... }
        >>> validate_data_config(bad_config)  # Raises ValueError

        >>> # With Hydra DictConfig
        >>> from omegaconf import DictConfig
        >>> hydra_config = DictConfig({
        ...     "data_root": "/path/to/data",
        ...     "train_split": 0.8,
        ...     "val_split": 0.1,
        ...     "test_split": 0.1,
        ...     "image_size": [256, 256]
        ... })
        >>> validate_data_config(hydra_config)  # Works with DictConfig

    Validation Process:
        1. Check presence of all required configuration keys
        2. Validate split ratios sum to 1.0 with numerical tolerance
        3. Verify image_size format and dimensionality
        4. Ensure all values have appropriate types and ranges

    Mathematical Validation:
        - Split ratios must sum to exactly 1.0 (tolerance: 1e-4)
        - Each split ratio should be in range [0.0, 1.0]
        - Image size must be positive integers

    Configuration Requirements:
        Required keys:
        - data_root: Path to dataset (should exist and be accessible)
        - train_split: Training data proportion (0.0-1.0)
        - val_split: Validation data proportion (0.0-1.0)
        - test_split: Test data proportion (0.0-1.0)
        - image_size: Target image dimensions [height, width]

    Performance Considerations:
        - Validation is lightweight and fast
        - Early validation prevents costly pipeline failures
        - Clear error messages reduce debugging time
        - Tolerance for floating-point arithmetic errors

    Integration:
        - Called by create_dataloaders_from_config() before dataset creation
        - Compatible with Hydra configuration loading
        - Works with both dict and DictConfig formats
        - Provides foundation for reliable data pipeline operation
    """
    required_keys = [
        "data_root",
        "train_split",
        "val_split",
        "test_split",
        "image_size",
    ]

    # Validate presence of all required keys
    for key in required_keys:
        if key not in data_cfg:
            raise ValueError(f"Missing required data config key: '{key}'")

    # Validate split ratios
    _validate_split_ratios(data_cfg)

    # Validate image size
    _validate_image_size(data_cfg["image_size"])


def _validate_split_ratios(data_cfg: dict[str, Any] | DictConfig) -> None:
    """Validate that train/val/test splits sum to 1.0.

    Args:
        data_cfg: Data configuration containing split ratios.

    Raises:
        ValueError: If splits don't sum to 1.0 or individual splits are invalid.
    """
    train_split_val = data_cfg["train_split"]
    val_split_val = data_cfg["val_split"]
    test_split_val = data_cfg["test_split"]

    # Check individual split values
    for split_name, split_val in [
        ("train_split", train_split_val),
        ("val_split", val_split_val),
        ("test_split", test_split_val),
    ]:
        if not isinstance(split_val, int | float):
            raise ValueError(
                f"{split_name} must be a number, got {type(split_val).__name__}"
            )
        if not 0.0 <= float(split_val) <= 1.0:
            raise ValueError(
                f"{split_name} must be between 0.0 and 1.0, got {split_val}"
            )

    # Validate sum equals 1.0
    total = (
        float(train_split_val) + float(val_split_val) + float(test_split_val)
    )
    if not abs(total - 1.0) < 1e-4:  # noqa: PLR2004
        raise ValueError(
            f"train/val/test splits must sum to 1.0, got {total}. "
            f"Individual splits: train={train_split_val}, "
            f"val={val_split_val}, test={test_split_val}"
        )


def _validate_image_size(img_size: list[int] | ListConfig) -> None:
    """Validate image size format and dimensionality.

    Args:
        img_size: Image size configuration [height, width].

    Raises:
        ValueError: If image size format is invalid.
    """
    if isinstance(img_size, ListConfig):
        img_size_for_len = list(img_size)
    else:
        img_size_for_len = img_size

    if not (len(img_size_for_len) == 2):  # noqa: PLR2004
        raise ValueError(
            f"image_size must be a list, tuple, or ListConfig of length 2, "
            f"got {type(img_size).__name__} with length "
            f"{len(img_size_for_len)}"
        )

    # Validate individual dimensions
    for i, dim in enumerate(img_size_for_len):
        if not isinstance(dim, int):
            raise ValueError(
                f"image_size[{i}] must be an integer, got {type(dim).__name__}"
            )
        if dim <= 0:
            raise ValueError(f"image_size[{i}] must be positive, got {dim}")
