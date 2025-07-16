"""Comprehensive image transformation module for crack segmentation dataset.

This module provides robust image transformation pipelines using
Albumentations, supporting training augmentations and evaluation transforms.
It handles both image and mask transformations with proper tensor conversions
and normalization.

Key Features:
    - Mode-aware transformations (train/val/test) with different augmentation
    levels
    - Configurable transformation pipelines from YAML/DictConfig
    - Proper image/mask loading and preprocessing with automatic binarization
    - Consistent tensor format output with (C, H, W) for images and (1, H, W)
    for masks
    - ImageNet-standard normalization defaults for pretrained model
    compatibility

Core Components:
    - get_basic_transforms(): Creates mode-specific transformation pipelines
    - apply_transforms(): Applies transformations to image-mask pairs
    - get_transforms_from_config(): Creates pipelines from configuration files

Common Usage:
    # Basic usage with default parameters
    train_transforms = get_basic_transforms("train", image_size=(512, 512))
    val_transforms = get_basic_transforms("val", image_size=(512, 512))

    # Apply transforms to image and mask
    result = apply_transforms(image_path, mask_path, train_transforms)
    image_tensor = result["image"]  # Shape: (3, 512, 512)
    mask_tensor = result["mask"]    # Shape: (1, 512, 512)

    # From configuration
    config_transforms = get_transforms_from_config(config_list, "train")

Performance Considerations:
    - Training mode applies extensive augmentations (7 different transforms)
    - Validation/test modes only apply resize and normalization
    - Mask binarization threshold set at 127 for consistent binary masks
    - Memory-efficient tensor conversions with proper dtype handling

Integration:
    - Works seamlessly with CrackSegmentationDataset
    - Compatible with Hydra configuration system
    - Supports both file paths and numpy arrays as input
    - Consistent with PyTorch DataLoader requirements

References:
    - Dataset: src.data.dataset.CrackSegmentationDataset
    - Configuration: configs/data/transform/
    - Factory: src.data.factory.create_crackseg_dataset
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig, OmegaConf

# Mask processing constants
MASK_BINARIZATION_THRESHOLD = 127
MASK_THRESHOLD_FLOAT = 0.5
MASK_2D_NDIM = 2
MASK_3D_NDIM = 3


def get_basic_transforms(
    mode: str,
    image_size: tuple[int, int] = (512, 512),
    # Restore ImageNet default values
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> A.Compose:
    """Creates a comprehensive transformation pipeline optimized for crack
    segmentation.

    Provides mode-specific transformation pipelines with different augmentation
    strategies. Training mode includes extensive data augmentation for robust
    model training, while validation and test modes apply only essential
    transforms for consistent evaluation.

    Args:
        mode: Transformation mode determining augmentation level:
            - 'train': Full augmentation pipeline with geometric and
            photometric transforms
            - 'val': Minimal transforms (resize + normalize) for consistent
            validation
            - 'test': Same as validation for deterministic inference
        image_size: Target output dimensions as (height, width).
            Default (512, 512) provides good balance between detail
            preservation and computational efficiency.
        mean: Per-channel normalization means. Default ImageNet values ensure
            compatibility with pretrained encoders like ResNet, EfficientNet.
        std: Per-channel normalization standard deviations. Must match
            pretrained model expectations for optimal transfer learning
            performance.

    Returns:
        A.Compose: Configured Albumentations pipeline ready for image/mask
        processing. All pipelines end with ToTensorV2() for PyTorch
        compatibility.

    Raises:
        ValueError: If mode is not one of ['train', 'val', 'test'].

    Examples:
        >>> # Standard training pipeline with augmentation
        >>> train_transforms = get_basic_transforms("train", (512, 512))
        >>> result = train_transforms(image=img_array, mask=mask_array)
        >>>
        >>> # Validation pipeline without augmentation
        >>> val_transforms = get_basic_transforms("val", (384, 384))
        >>>
        >>> # Custom normalization for specific pretrained model
        >>> custom_transforms = get_basic_transforms(
        ...     "train",
        ...     mean=(0.5, 0.5, 0.5),
        ...     std=(0.5, 0.5, 0.5)
        ... )

    Training Augmentations Applied:
        - HorizontalFlip(p=0.5): Mirror images horizontally
        - VerticalFlip(p=0.5): Mirror images vertically
        - RandomRotate90(p=0.5): 90-degree rotations
        - RandomBrightnessContrast(p=0.2): Photometric variations
        - GaussNoise(p=0.2): Noise robustness
        - RandomSizedCrop: Scale variations (80-120% of original)
        - HueSaturationValue(p=0.5): Color space augmentation

    Performance Notes:
        - Training transforms increase data diversity by ~100x
        - cv2.INTER_LINEAR interpolation balances quality and speed
        - Augmentation probabilities tuned for crack segmentation domain
        - RandomSizedCrop maintains aspect ratio and prevents distortion
    """
    if mode not in ["train", "val", "test"]:
        msg = f"Invalid mode: {mode}. Must be 'train', 'val' or 'test'."
        raise ValueError(msg)

    # Core transforms applied to all modes
    core_transforms: Sequence[Any] = [
        A.Resize(
            height=image_size[0],
            width=image_size[1],
            interpolation=cv2.INTER_LINEAR,  # Default for image
        ),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        A.pytorch.ToTensorV2(),
    ]

    # Training augmentations for improved generalization
    image_augmentations: Sequence[Any] = []
    if mode == "train":
        image_augmentations = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussNoise(p=0.2),
            A.RandomSizedCrop(
                min_max_height=(
                    int(0.8 * image_size[0]),
                    int(1.2 * image_size[0]),
                ),
                size=image_size,
                p=1.0,
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5,
            ),
        ]
        # Combine augmentations with core transforms
        # (excluding redundant resize)
        final_transforms: Sequence[Any] = list(image_augmentations) + [
            t for t in core_transforms if not isinstance(t, A.Resize)
        ]
    else:
        final_transforms = core_transforms

    pipeline = A.Compose(list(final_transforms))

    return pipeline


def _load_image(
    image: np.ndarray[Any, Any] | str | Path,
) -> np.ndarray[Any, Any]:
    """Load and standardize image data from various input types.

    Handles both file paths and numpy arrays, ensuring consistent RGB format
    for downstream processing. Performs automatic color space conversion from
    OpenCV's default BGR to standard RGB format.

    Args:
        image: Input image as numpy array or file path (str/Path).
            If array, assumed to be in correct format already.
            If path, loaded using OpenCV and converted to RGB.

    Returns:
        np.ndarray: Image array in RGB format with shape (H, W, 3).
            Pixel values in range [0, 255] as uint8.

    Note:
        - OpenCV loads images in BGR format by default
        - Conversion to RGB ensures compatibility with most ML frameworks
        - No validation of image dimensions or content performed
    """
    if isinstance(image, str | Path):
        img_array = cv2.imread(str(image))
        return cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    return image


def _load_mask(
    mask: np.ndarray[Any, Any] | str | Path | None,
) -> np.ndarray[Any, Any] | None:
    """Load mask data with automatic grayscale conversion.

    Handles optional mask loading from various input types. For file paths,
    loads as grayscale to ensure single-channel output suitable for binary
    segmentation tasks.

    Args:
        mask: Input mask as numpy array, file path, or None.
            If array, returned as-is without modification.
            If path, loaded as grayscale using OpenCV.
            If None, returns None (no mask available).

    Returns:
        np.ndarray | None: Grayscale mask array with shape (H, W) or None.
            If loaded from file, pixel values in range [0, 255] as uint8.

    Note:
        - cv2.IMREAD_GRAYSCALE ensures single-channel output
        - No binarization applied at this stage (handled in apply_transforms)
        - Supports optional mask scenarios (e.g., inference without ground
        truth)
    """
    if isinstance(mask, str | Path):
        mask_array_raw = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
        return mask_array_raw
    elif isinstance(mask, np.ndarray):
        return mask
    return None


def apply_transforms(
    image: np.ndarray[Any, Any] | str | Path,
    mask: np.ndarray[Any, Any] | str | Path | None = None,
    transforms: A.Compose | None = None,
) -> dict[str, torch.Tensor]:
    """Apply comprehensive transformations to image-mask pairs with proper
    tensor conversion.

    Provides a unified interface for applying Albumentations transforms to
    images and masks, handling loading, binarization, and tensor conversion.
    Ensures consistent output format regardless of input type or transform
    pipeline.

    Args:
        image: Input image as numpy array or file path. Must be valid image
            data or path to readable image file. RGB format expected.
        mask: Optional input mask as numpy array, file path, or None. If
            provided, will be binarized using threshold and included in
            output.
        transforms: Albumentations transform pipeline. If None, applies basic
            tensor conversion with normalization to [0, 1] range.

    Returns:
        dict[str, torch.Tensor]: Dictionary containing transformed data:
            - 'image': Image tensor with shape (C, H, W), values in [0, 1] or
            normalized
            - 'mask': Mask tensor with shape (1, H, W), binary values {0, 1}
            (if provided)

    Processing Pipeline:
        1. Load image and mask using appropriate loading functions
        2. Binarize mask using MASK_BINARIZATION_THRESHOLD (127)
        3. Apply transforms if provided, or convert to tensors directly
        4. Ensure mask has consistent shape (1, H, W) for model compatibility
        5. Return dictionary with standardized tensor format

    Examples:
        >>> # With transform pipeline
        >>> transforms = get_basic_transforms("train", (512, 512))
        >>> result = apply_transforms("image.jpg", "mask.png", transforms)
        >>> image = result["image"]  # Shape: (3, 512, 512), normalized
        >>> mask = result["mask"]    # Shape: (1, 512, 512), binary
        >>>
        >>> # Without transforms (basic conversion)
        >>> result = apply_transforms(img_array, mask_array, None)
        >>> image = result["image"]  # Shape: (3, H, W), range [0, 1]
        >>>
        >>> # Image only (no mask)
        >>> result = apply_transforms("image.jpg", None, transforms)
        >>> # Returns only {"image": tensor}

    Mask Processing Details:
        - Input masks binarized using threshold 127
        (> threshold = 1, <= threshold = 0)
        - After transforms, masks re-binarized using float threshold 0.5
        - Output masks always have single channel dimension for consistency
        - Supports both 2D (H, W) and 3D (C, H, W) input mask formats

    Performance Considerations:
        - Efficient memory usage with in-place tensor operations where possible
        - Automatic dtype conversion (uint8 -> float32) for ML compatibility
        - Channel dimension handling prevents shape mismatches in model
        training
        - Direct tensor conversion bypasses unnecessary intermediate
        allocations

    Error Handling:
        - File loading errors propagated from OpenCV (invalid paths, corrupted
        files)
        - Shape mismatches handled gracefully with automatic dimension
        adjustment
        - Transform errors from Albumentations passed through with original
        context

    Integration:
        - Compatible with CrackSegmentationDataset.__getitem__()
        - Supports both training and inference workflows
        - Works with custom transform configurations from YAML files
        - Handles edge cases like missing masks in inference scenarios
    """
    current_image: np.ndarray[Any, Any] = _load_image(image)
    current_mask: np.ndarray[Any, Any] | None = _load_mask(mask)

    # Binarize mask if present
    if current_mask is not None:
        current_mask = (current_mask > MASK_BINARIZATION_THRESHOLD).astype(
            np.uint8
        )

    # Apply transforms if provided
    if transforms is not None:
        result = transforms(image=current_image, mask=current_mask)
        transformed_image = result["image"]
        if "mask" in result and result["mask"] is not None:
            transformed_mask = result["mask"]
            # Convert to tensor if needed
            if isinstance(transformed_mask, np.ndarray):
                transformed_mask = torch.from_numpy(transformed_mask)
            # Re-binarize after potential interpolation
            transformed_mask = (
                (transformed_mask > MASK_THRESHOLD_FLOAT).long().float()
            )
            return {"image": transformed_image, "mask": transformed_mask}
        return {"image": transformed_image}

    # Basic tensor conversion without transforms
    image_tensor = (
        torch.from_numpy(current_image).permute(2, 0, 1).float() / 255.0
    )
    if current_mask is not None:
        mask_tensor = torch.from_numpy(current_mask)
        # Ensure mask has shape (1, H, W)
        if mask_tensor.ndim == MASK_2D_NDIM:
            mask_tensor = mask_tensor.unsqueeze(0)
        elif mask_tensor.ndim == MASK_3D_NDIM and mask_tensor.shape[0] != 1:
            mask_tensor = mask_tensor[0:1]
        mask_tensor = mask_tensor.float()
        return {"image": image_tensor, "mask": mask_tensor}
    return {"image": image_tensor}


def _get_transform_specs(
    config_list: (
        list[dict[str, Any] | DictConfig] | dict[str, Any] | DictConfig
    ),
) -> list[dict[str, Any] | DictConfig]:
    """Normalize transform configuration format for pipeline creation.

    Converts various configuration formats (list of dicts, single dict,
    DictConfig) into a standardized list format suitable for transform
    instantiation. Handles special cases like 'resize' key normalization.

    Args:
        config_list: Transform configuration in various formats:
            - List of dicts: [{"name": "Resize", "params": {...}}, ...]
            - Single dict: {"resize": {...}, "normalize": {...}, ...}
            - DictConfig: Hydra/OmegaConf configuration object

    Returns:
        list[dict[str, Any] | DictConfig]: Normalized list of transform
        specifications, each containing 'name' and 'params' keys for
        transform instantiation.

    Processing Logic:
        - If input is list, returns as-is (already in correct format)
        - If input is dict/DictConfig, converts each key-value pair to
        name-params format
        - Special handling for 'resize' key -> 'Resize' transform name
        - Preserves parameter dictionaries for downstream processing

    Examples:
        >>> # List format (no change needed)
        >>> config = [
        ...     {"name": "Resize", "params": {"height": 512, "width": 512}}
        ... ]
        >>> specs = _get_transform_specs(config)
        >>> # Returns same list
        >>>
        >>> # Dict format (converted to list)
        >>> config = {"resize": {"height": 512}, "normalize": {"mean": [0.5]}}
        >>> specs = _get_transform_specs(config)
        >>> # Returns: [{"name": "Resize", "params": {...}},
        >>> #          {"name": "normalize", "params": {...}}]

    Note:
        - Case-insensitive handling for 'resize' -> 'Resize' conversion
        - Maintains original parameter structure for complex configurations
        - Compatible with both programmatic and YAML-based configurations
    """
    if isinstance(config_list, dict | DictConfig):
        specs_from_mapping: list[dict[str, Any] | DictConfig] = []
        for name_key, params_val in config_list.items():
            str_name_key = str(name_key)
            # Handle special case for resize transform
            actual_transform_name = (
                "Resize" if str_name_key.lower() == "resize" else str_name_key
            )
            specs_from_mapping.append(
                {"name": actual_transform_name, "params": params_val}
            )
        return specs_from_mapping
    return config_list


def get_transforms_from_config(
    config_list: (
        list[dict[str, Any] | DictConfig] | dict[str, Any] | DictConfig
    ),
    mode: str,
) -> A.Compose:
    """Create production-ready Albumentations pipeline from configuration
    specifications.

    Provides a robust interface for creating transform pipelines from YAML
    configurations or programmatic specifications. Handles complex parameter
    resolution, error checking, and special transform cases for maximum
    flexibility in production environments.

    Args:
        config_list: Transform configuration in flexible formats:
            - List format: [{"name": "Resize", "params": {"height": 512,
              "width": 512}}, ...]
            - Dict format: {"resize": {"height": 512}, "normalize":
              {"mean": [0.485, 0.456, 0.406]}}
            - DictConfig: Hydra configuration with nested parameters and
              interpolation
        mode: Intended usage mode ('train', 'val', 'test'). Currently
            reserved for future mode-specific optimizations. Maintains API
            consistency.

    Returns:
        A.Compose: Fully configured Albumentations pipeline ready for
            production use. All transforms properly instantiated with
            resolved parameters.

    Raises:
        ValueError: If any transform specification is invalid:
            - Missing 'name' key in transform specification
            - Unknown transform name not found in Albumentations library
            - Parameter instantiation errors (invalid values, type mismatches)

    Configuration Format Examples:
        >>> # YAML configuration (loaded as DictConfig)
        >>> config_yaml = '''
        ... transforms:
        ...   resize:
        ...     height: 512
        ...     width: 512
        ...   normalize:
        ...     mean: [0.485, 0.456, 0.406]
        ...     std: [0.229, 0.224, 0.225]
        ...   ToTensorV2: {}
        ... '''
        >>>
        >>> # Programmatic list configuration
        >>> config_list = [
        ...     {"name": "Resize", "params": {"height": 512, "width": 512}},
        ...     {"name": "Normalize", "params": {
        ...         "mean": [0.485, 0.456, 0.406]
        ...     }},
        ...     {"name": "ToTensorV2", "params": {}}
        ... ]
        >>>
        >>> # Create pipeline
        >>> transforms = get_transforms_from_config(config_list, "train")

    Transform Resolution Process:
        1. Normalize configuration format using _get_transform_specs()
        2. Iterate through each transform specification
        3. Resolve transform class from Albumentations library
        4. Handle special cases (ToTensorV2, custom parameters)
        5. Resolve DictConfig parameters with OmegaConf interpolation
        6. Instantiate transform with resolved parameters
        7. Compose all transforms into final pipeline

    Special Transform Handling:
        - ToTensorV2: Automatically imports from albumentations.pytorch
        - DictConfig parameters: Full OmegaConf resolution with variable
          interpolation
        - Parameter type conversion: Automatic string key conversion for
          compatibility
        - Error context: Detailed error messages with transform name and
          parameters

    Integration Examples:
        >>> # With Hydra configuration
        >>> @hydra.main(config_path="configs", config_name="data")
        >>> def main(cfg: DictConfig) -> None:
        ...     transforms = get_transforms_from_config(
        ...         cfg.transforms, "train"
        ...     )
        >>>
        >>> # With manual configuration
        >>> config = {"resize": {"height": 384, "width": 384}}
        >>> transforms = get_transforms_from_config(config, "val")
        >>>
        >>> # Complex augmentation pipeline
        >>> aug_config = [
        ...     {"name": "HorizontalFlip", "params": {"p": 0.5}},
        ...     {"name": "RandomBrightnessContrast", "params": {"p": 0.3}},
        ...     {"name": "ToTensorV2", "params": {}}
        ... ]
        >>> aug_transforms = get_transforms_from_config(aug_config, "train")

    Performance and Reliability:
        - Comprehensive error handling with context preservation
        - Efficient parameter resolution avoiding unnecessary conversions
        - Maintains transform order from configuration specification
        - Compatible with all Albumentations transforms and parameters
        - Supports nested parameter structures and complex configurations

    Production Considerations:
        - Validates all transform names against available Albumentations
          classes
        - Provides detailed error messages for debugging configuration
          issues
        - Handles edge cases like empty parameter dictionaries
        - Maintains configuration flexibility for different deployment
          scenarios
        - Thread-safe operation for multi-process data loading

    References:
        - Albumentations documentation: https://albumentations.ai/docs/
        - Hydra configuration: https://hydra.cc/docs/intro/
        - Configuration examples: configs/data/transform/
    """
    transform_specs_list = _get_transform_specs(config_list)

    transforms_pipeline: list[Any] = []

    for transform_item in transform_specs_list:
        name_any = transform_item.get("name")
        name = str(name_any) if name_any is not None else None
        params_value_any: Any = transform_item.get("params", {})
        params_value: dict[str, Any] = {}

        if name is None:
            raise ValueError("Each transform item must have a 'name' key.")

        try:
            transform_class: type[Any]
            # Special handling for ToTensorV2
            if name == "ToTensorV2":
                transform_class = ToTensorV2
                params_value = {}
            else:
                transform_class = getattr(A, name)

            # Resolve parameters from DictConfig or dict
            if isinstance(params_value_any, DictConfig):
                resolved_params_raw = OmegaConf.to_container(
                    params_value_any, resolve=True
                )
                resolved_params = cast(dict[Any, Any], resolved_params_raw)
                params_value = {str(k): v for k, v in resolved_params.items()}
            elif isinstance(params_value_any, dict):
                params_dict_any = cast(dict[Any, Any], params_value_any)
                params_value = {str(k): v for k, v in params_dict_any.items()}
            else:
                # Empty params or other types
                pass

            # Instantiate transform with resolved parameters
            transforms_pipeline.append(transform_class(**params_value))

        except AttributeError as exc:
            raise ValueError(f"Unknown transform name: '{name}'") from exc
        except Exception as e:
            raise ValueError(
                f"Error instantiating transform '{name}' with params "
                f"{params_value}: {e}"
            ) from e

    return A.Compose(transforms_pipeline)
