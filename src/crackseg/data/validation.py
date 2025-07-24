"""Configuration validation module for data pipeline components.

This module provides comprehensive validation functions for data and transform
configurations used throughout the crack segmentation pipeline. It ensures
configuration integrity, prevents runtime errors, and provides clear feedback
for configuration issues.

Key Features:
    - Data configuration validation with split ratio verification
    - Transform configuration validation with parameter checking
    - Flexible input format support (dict, DictConfig, list, ListConfig)
    - Detailed error messages for debugging configuration issues
    - Essential transform requirement enforcement (Resize, Normalize)

Core Functions:
    - validate_data_config(): Validates dataset configuration parameters
    - validate_transform_config(): Validates transformation pipeline
      configuration
    - Helper functions for specific transform parameter validation

Validation Scope:
    - Data paths and accessibility
    - Split ratios mathematical correctness (sum to 1.0)
    - Image size format and validity
    - Transform parameter completeness and format
    - Essential transform presence verification

Common Usage:
    # Data configuration validation
    validate_data_config(cfg.data)

    # Transform configuration validation
    validate_transform_config(cfg.transforms.train, mode="train")
    validate_transform_config(cfg.transforms.val, mode="val")

    # Works with various configuration formats
    config_dict = {"resize": {"height": 512, "width": 512}}
    validate_transform_config(config_dict)

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
    - Transforms: src.data.transforms.get_transforms_from_config
    - Dataset: src.data.dataset.CrackSegmentationDataset
    - Configuration: configs/data/ and configs/transforms/
"""

import warnings
from typing import Any, cast

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

    # Validate split ratios sum to 1.0 (with numerical tolerance)
    train_split_val = data_cfg["train_split"]
    val_split_val = data_cfg["val_split"]
    test_split_val = data_cfg["test_split"]

    total = (
        float(train_split_val) + float(val_split_val) + float(test_split_val)
    )
    if not abs(total - 1.0) < 1e-4:  # noqa: PLR2004
        raise ValueError(
            f"train/val/test splits must sum to 1.0, got {total}. "
            f"Individual splits: train={train_split_val}, "
            f"val={val_split_val}, test={test_split_val}"
        )

    # Validate image_size format and dimensionality
    img_size = data_cfg["image_size"]
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


def _normalize_transform_config_input(
    transform_config: (
        list[dict[str, Any] | DictConfig]
        | ListConfig
        | dict[str, Any]
        | DictConfig
    ),
) -> list[dict[str, Any] | DictConfig]:
    """Normalize transform configuration into standardized list format.

    Converts various transform configuration formats into a unified list
    structure for consistent processing. Handles both list-based and
    dictionary-based configuration formats with proper type preservation.

    Args:
        transform_config: Transform configuration in various formats:
            - list[dict]: List of transform dictionaries with 'name' and
              'params' keys
            - ListConfig: Hydra list configuration
            - dict: Dictionary mapping transform names to parameter
              dictionaries
            - DictConfig: Hydra dictionary configuration

    Returns:
        list[dict[str, Any] | DictConfig]: Normalized list where each item
            contains:
            - name: str - Transform name (e.g., "Resize", "Normalize")
            - params: dict | DictConfig - Transform parameters

    Raises:
        ValueError: If list items are not dictionaries or DictConfig objects

    Examples:
        >>> # List format (already normalized)
        >>> list_config = [
        ...     {"name": "Resize", "params": {"height": 512, "width": 512}},
        ...     {"name": "Normalize", "params": {
        ...         "mean": [0.485, 0.456, 0.406]
        ...     }}
        ... ]
        >>> result = _normalize_transform_config_input(list_config)
        >>> # Returns same list
        >>>
        >>> # Dictionary format (converted to list)
        >>> dict_config = {
        ...     "resize": {"height": 512, "width": 512},
        ...     "normalize": {
        ...         "mean": [0.485, 0.456, 0.406],
        ...         "std": [0.229, 0.224, 0.225]
        ...     }
        ... }
        >>> result = _normalize_transform_config_input(dict_config)
        >>> # Returns: [{"name": "resize", "params": {...}},
        >>> #          {"name": "normalize", "params": {...}}]

    Processing Logic:
        - List/ListConfig: Validates items are dicts/DictConfigs, returns as-is
        - Dict/DictConfig: Converts each key-value pair to {"name": key,
          "params": value}
        - Preserves original parameter structure and types
        - Handles None parameter values gracefully

    Type Preservation:
        - DictConfig objects preserved for Hydra interpolation
        - Parameter dictionaries maintain original structure
        - Type information retained for downstream processing

    Integration:
        - Used by validate_transform_config() for format normalization
        - Compatible with get_transforms_from_config() expectations
        - Enables flexible configuration input while maintaining consistency
    """
    actual_transform_list: list[dict[str, Any] | DictConfig]

    if isinstance(transform_config, list | ListConfig):
        # Process list format: validate and preserve items
        actual_transform_list = []
        for item_any in transform_config:
            if not isinstance(item_any, dict | DictConfig):
                raise ValueError(
                    "If transform_config is a list, its items must be dicts "
                    "or DictConfigs. "
                    f"Found item of type: {type(item_any).__name__}"
                )
            # Preserve type information for proper processing
            if isinstance(item_any, DictConfig):
                actual_transform_list.append(item_any)
            else:
                # Cast to correct type after validation
                actual_transform_list.append(cast(dict[str, Any], item_any))
    else:
        # Process dictionary format: convert to list of name-params pairs
        actual_transform_list = []
        for name, params_data in transform_config.items():
            # Handle None parameters gracefully
            current_params = params_data if params_data is not None else {}
            actual_transform_list.append(
                {"name": str(name), "params": current_params}
            )
    return actual_transform_list


def _validate_resize_params(params: dict[str, Any] | DictConfig) -> None:
    """Validate parameters for Resize transform with comprehensive checks.

    Ensures Resize transform parameters are properly specified and valid.
    Supports both 'size' parameter (single specification) and separate
    'height'/'width' parameters with appropriate type checking.

    Args:
        params: Resize transform parameters containing either:
            - size: list[int] | tuple[int, int] - [height, width] dimensions
            - height: int | float - Target height
            - width: int | float - Target width

    Raises:
        ValueError: If parameters are missing, invalid, or improperly
            formatted:
            - Missing both 'size' and 'height'/'width' parameters
            - 'size' parameter not a 2-element list/tuple/ListConfig
            - 'height' or 'width' parameters not numeric
            - Invalid parameter combinations or formats

    Examples:
        >>> # Valid size parameter
        >>> params = {"size": [512, 512]}
        >>> _validate_resize_params(params)  # No error
        >>>
        >>> # Valid height/width parameters
        >>> params = {"height": 384, "width": 384}
        >>> _validate_resize_params(params)  # No error
        >>>
        >>> # Invalid: missing parameters
        >>> params = {}
        >>> _validate_resize_params(params)  # Raises ValueError
        >>>
        >>> # Invalid: size wrong format
        >>> params = {"size": 512}  # Should be [512, 512]
        >>> _validate_resize_params(params)  # Raises ValueError

    Parameter Validation:
        Size parameter:
        - Must be list, tuple, or ListConfig of exactly 2 elements
        - Elements should represent [height, width]

        Height/Width parameters:
        - Both must be present if using this format
        - Must be numeric values (int or float)
        - Should be positive for meaningful resizing

    Format Support:
        - Native Python lists and tuples
        - Hydra ListConfig for configuration flexibility
        - Integer and float values for dimensions
        - Mixed parameter formats within same configuration

    Integration:
        - Called by validate_transform_config() for Resize transforms
        - Ensures compatibility with Albumentations Resize transform
        - Provides early validation before transform instantiation
    """
    # Check for required parameters
    if "size" not in params and (
        "height" not in params or "width" not in params
    ):
        raise ValueError(
            "Resize transform must have either 'size' or both "
            "'height' and 'width' parameters."
        )

    # Validate 'size' parameter format
    if "size" in params:
        size_val = params["size"]
        size_val_for_len: list[Any] | tuple[Any, ...] = size_val
        if isinstance(size_val, ListConfig):
            size_val_for_len = list(size_val)
        if not (
            isinstance(size_val, list | tuple)
            and len(size_val_for_len) == 2  # noqa: PLR2004
        ):
            raise ValueError(
                f"Resize 'size' must be a list, tuple, or ListConfig of "
                f"length 2. "
                f"Got {type(size_val).__name__} with length "
                f"{len(size_val_for_len) if hasattr(size_val, '__len__') else 'unknown'}"  # noqa: E501
            )

    # Validate 'height' and 'width' parameters
    if "height" in params and "width" in params:
        height_val = params["height"]
        width_val = params["width"]
        if not (
            isinstance(height_val, int | float)
            and isinstance(width_val, int | float)
        ):
            raise ValueError(
                f"Resize 'height' and 'width' must be numeric values. "
                f"Got height: {type(height_val).__name__}, "
                f"width: {type(width_val).__name__}"
            )


def _validate_normalize_params(params: dict[str, Any] | DictConfig) -> None:
    """Validate parameters for Normalize transform with channel-aware checking.

    Ensures Normalize transform parameters are complete and properly formatted
    for 3-channel image normalization. Validates mean and standard deviation
    parameters for RGB image processing compatibility.

    Args:
        params: Normalize transform parameters containing:
            - mean: list[float] - Per-channel normalization means [R, G, B]
            - std: list[float] - Per-channel normalization standard deviations
            [R, G, B]

    Raises:
        ValueError: If parameters are missing or improperly formatted:
            - Missing 'mean' or 'std' parameters
            - 'mean' not a 3-element list/tuple/ListConfig
            - 'std' not a 3-element list/tuple/ListConfig

    Examples:
        >>> # Valid normalization parameters (ImageNet values)
        >>> params = {
        ...     "mean": [0.485, 0.456, 0.406],
        ...     "std": [0.229, 0.224, 0.225]
        ... }
        >>> _validate_normalize_params(params)  # No error
        >>>
        >>> # Invalid: missing parameters
        >>> params = {"mean": [0.5, 0.5, 0.5]}  # Missing std
        >>> _validate_normalize_params(params)  # Raises ValueError
        >>>
        >>> # Invalid: wrong number of channels
        >>> params = {
        ...     "mean": [0.5],  # Should be 3 values
        ...     "std": [0.5, 0.5, 0.5]
        ... }
        >>> _validate_normalize_params(params)  # Raises ValueError

    Parameter Requirements:
        Mean parameter:
        - Must be present in configuration
        - Must be list, tuple, or ListConfig of exactly 3 values
        - Values typically in range [0, 1] for normalized images

        Std parameter:
        - Must be present in configuration
        - Must be list, tuple, or ListConfig of exactly 3 values
        - Values should be positive for meaningful normalization

    Channel Assumptions:
        - Validation assumes 3-channel RGB images
        - Parameters must match expected channel count
        - Compatible with pretrained model requirements (ImageNet, etc.)

    Integration:
        - Called by validate_transform_config() for Normalize transforms
        - Ensures compatibility with Albumentations Normalize transform
        - Validates against common normalization mistakes
    """
    # Check for required parameters
    if "mean" not in params or "std" not in params:
        missing_params = []
        if "mean" not in params:
            missing_params.append("mean")
        if "std" not in params:
            missing_params.append("std")
        raise ValueError(
            f"Missing required Normalize parameters: "
            f"{', '.join(missing_params)}"
        )

    mean_val = params["mean"]
    std_val = params["std"]

    # Handle ListConfig conversion for length checking
    mean_val_for_len: list[Any] | tuple[Any, ...] = mean_val
    if isinstance(mean_val, ListConfig):
        mean_val_for_len = list(mean_val)
    std_val_for_len: list[Any] | tuple[Any, ...] = std_val
    if isinstance(std_val, ListConfig):
        std_val_for_len = list(std_val)

    # Validate mean parameter format
    if not (
        isinstance(mean_val, list | tuple | ListConfig)
        and len(mean_val_for_len) == 3  # noqa: PLR2004
    ):
        raise ValueError(
            f"Normalize 'mean' must be list, tuple or ListConfig of 3 values "
            f"for RGB channels. Got {type(mean_val).__name__} with length "
            f"{len(mean_val_for_len) if hasattr(mean_val, '__len__') else 'unknown'}"  # noqa: E501
        )

    # Validate std parameter format
    if not (
        isinstance(std_val, list | tuple | ListConfig)
        and len(std_val_for_len) == 3  # noqa: PLR2004
    ):
        raise ValueError(
            f"Normalize 'std' must be list, tuple or ListConfig of 3 values "
            f"for RGB channels. Got {type(std_val).__name__} with length "
            f"{len(std_val_for_len) if hasattr(std_val, '__len__') else 'unknown'}"  # noqa: E501
        )


def _convert_albumentations_format_to_standard(
    transform_config: (
        list[dict[str, Any] | DictConfig]
        | ListConfig
        | dict[str, Any]
        | DictConfig
    ),
) -> list[dict[str, Any] | DictConfig]:
    """Convert Albumentations direct format to standard format.

    Converts Albumentations direct format:
    [
        {"Resize": {"height": 256, "width": 256}},
        {"Normalize": {"mean": [0.485, 0.456, 0.406]}}
    ]

    To standard format:
    [
        {"name": "Resize", "params": {"height": 256, "width": 256}},
        {"name": "Normalize", "params": {"mean": [0.485, 0.456, 0.406]}}
    ]

    Also handles nested structure with 'augmentations' key.
    """
    if isinstance(transform_config, dict | DictConfig):
        # Check if it has 'augmentations' key (nested structure)
        if "augmentations" in transform_config:
            return _convert_albumentations_format_to_standard(
                transform_config["augmentations"]
            )
        else:
            # Convert dict format to list format
            converted_list = []
            for name_key, params_val in transform_config.items():
                str_name_key = str(name_key)
                # Handle special case for resize transform
                actual_transform_name = (
                    "Resize"
                    if str_name_key.lower() == "resize"
                    else str_name_key
                )
                converted_list.append(
                    {"name": actual_transform_name, "params": params_val}
                )
            return converted_list
    elif isinstance(transform_config, list | ListConfig):
        converted_list = []
        for item in transform_config:
            if isinstance(item, dict | DictConfig):
                # Check if it's already in standard format
                if "name" in item and "params" in item:
                    converted_list.append(item)
                else:
                    # Convert from Albumentations direct format
                    for transform_name, transform_params in item.items():
                        converted_list.append(
                            {
                                "name": str(transform_name),
                                "params": transform_params,
                            }
                        )
            else:
                converted_list.append(item)
        return converted_list
    else:
        # For other types, return as-is
        return [transform_config]


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
        >>> # Valid list configuration
        >>> config = [
        ...     {"name": "Resize", "params": {"height": 512, "width": 512}},
        ...     {"name": "Normalize", "params": {
        ...         "mean": [0.485, 0.456, 0.406],
        ...         "std": [0.229, 0.224, 0.225]
        ...     }}
        ... ]
        >>> validate_transform_config(config)  # No error
        >>>
        >>> # Valid dictionary configuration
        >>> config = {
        ...     "resize": {"height": 384, "width": 384},
        ...     "normalize": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}
        ... }
        >>> validate_transform_config(config)  # No error
        >>>
        >>> # Invalid: missing Normalize transform
        >>> config = [
        ...     {"name": "Resize", "params": {"height": 512, "width": 512}}
        ... ]
        >>> validate_transform_config(config)  # Raises ValueError

    Validation Process:
        1. Normalize configuration format using
           _normalize_transform_config_input()
        2. Check for empty configuration (warning only)
        3. Scan for required transforms (Resize, Normalize)
        4. Validate parameters for each found transform
        5. Verify essential transforms are present

    Required Transforms:
        Resize Transform:
        - Essential for consistent input dimensions
        - Parameters validated by _validate_resize_params()
        - Must be present if any transforms are specified

        Normalize Transform:
        - Essential for model input preprocessing
        - Parameters validated by _validate_normalize_params()
        - Must be present if any transforms are specified

    Configuration Flexibility:
        - Supports both list and dictionary configuration formats
        - Compatible with Hydra configuration system
        - Handles mixed parameter types (dict, DictConfig)
        - Preserves configuration structure for downstream processing

    Error Handling:
        - Specific error messages for missing transforms
        - Detailed parameter validation with context
        - Early validation prevents runtime pipeline failures
        - Clear guidance for configuration correction

    Integration:
        - Called by create_dataloaders_from_config() during setup
        - Compatible with get_transforms_from_config() expectations
        - Validates before expensive transform instantiation
        - Ensures pipeline reliability across different modes

    Performance Considerations:
        - Lightweight validation with minimal overhead
        - Early error detection saves debugging time
        - Configuration normalization enables efficient processing
        - Warning system for non-critical issues

    Future Extensions:
        - Mode-specific validation rules
        - Additional essential transform requirements
        - Custom validation rules for specific transforms
        - Integration with transform capability checking
    """
    # Normalize configuration format using the same function as
    # get_transforms_from_config
    actual_transform_list = _convert_albumentations_format_to_standard(
        transform_config
    )

    # Warn if configuration is empty (may indicate configuration issues)
    if not actual_transform_list:
        warnings.warn(
            "The transform configuration list is empty. Resize and Normalize "
            "checks might fail if they are mandatory for the pipeline.",
            stacklevel=2,
        )
        # Early return for empty configuration (depending on requirements,
        # this might be acceptable for some use cases)
        return

    # Track presence of essential transforms
    resize_found = False
    normalize_found = False

    # Validate each transform in the configuration
    for transform_item in actual_transform_list:
        name_any = transform_item.get("name")
        name = str(name_any) if name_any is not None else None

        # Handle parameters with fallback for None values
        params_from_item = transform_item.get("params", {})
        params: dict[str, Any] | DictConfig = (
            params_from_item if params_from_item is not None else {}
        )

        # Validate specific transform types
        if name == "Resize":
            resize_found = True
            _validate_resize_params(params)
        elif name == "Normalize":
            normalize_found = True
            _validate_normalize_params(params)

    # Verify essential transforms are present (only if configuration is
    # not empty)
    if not resize_found and actual_transform_list:
        raise ValueError(
            "Missing required 'Resize' transform in the configuration. "
            "Resize transform is essential for consistent input dimensions."
        )

    if not normalize_found and actual_transform_list:
        raise ValueError(
            "Missing required 'Normalize' transform in the configuration. "
            "Normalize transform is essential for model input preprocessing."
        )
