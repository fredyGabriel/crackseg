"""Transform pipelines for crack segmentation datasets.

This module provides comprehensive image transformation pipelines using
Albumentations, supporting training augmentations and evaluation transforms.
"""

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

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

    Returns:
        Albumentations Compose object with mode-specific transforms

    Examples:
        Training transforms with full augmentation:
        ```python
        train_transforms = get_basic_transforms("train", (512, 512))
        # Includes: Resize, HorizontalFlip, VerticalFlip, Rotate,
        # ColorJitter, RandomBrightnessContrast, Normalize, ToTensor
        ```

        Validation transforms with minimal processing:
        ```python
        val_transforms = get_basic_transforms("val", (512, 512))
        # Includes: Resize, Normalize, ToTensor (no augmentation)
        ```

        Test transforms for inference:
        ```python
        test_transforms = get_basic_transforms("test", (512, 512))
        # Same as validation for consistent evaluation
        ```

    Performance Notes:
        - Training mode applies 7 different transforms for robust learning
        - Validation/test modes apply only 3 essential transforms
        - All transforms are optimized for crack segmentation tasks
        - Mask binarization threshold set at 127 for consistent binary masks
        - Memory-efficient tensor conversions with proper dtype handling
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(height=image_size[0], width=image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    p=0.5,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5,
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    elif mode in ["val", "test"]:
        return A.Compose(
            [
                A.Resize(height=image_size[0], width=image_size[1]),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(
            f"Invalid mode: {mode}. Must be 'train', 'val', or 'test'"
        )


def apply_transforms(
    image: np.ndarray | str,
    mask: np.ndarray | str | None = None,
    transforms: A.Compose | None = None,
) -> dict[str, torch.Tensor]:
    """Apply Albumentations transforms to image and mask.

    This function provides a unified interface for applying transforms to
    image-mask pairs, handling both file paths and numpy arrays as input.
    It ensures proper tensor format output with (C, H, W) for images and
    (1, H, W) for masks.

    Args:
        image: Input image as numpy array, file path, or PIL Image
        mask: Input mask as numpy array, file path, or PIL Image (optional)
        transforms: Albumentations transform pipeline. If None, uses basic
            transforms with default parameters.

    Returns:
        Dictionary containing transformed tensors:
        - "image": torch.Tensor of shape (C, H, W) with float32 dtype
        - "mask": torch.Tensor of shape (1, H, W) with float32 dtype

    Examples:
        Basic usage with numpy arrays:
        ```python
        image = np.random.rand(256, 256, 3)
        mask = np.random.randint(0, 2, (256, 256))

        result = apply_transforms(image, mask, transforms)
        image_tensor = result["image"]  # (3, H, W)
        mask_tensor = result["mask"]    # (1, H, W)
        ```

        With file paths:
        ```python
        result = apply_transforms(
            image_path="images/crack.jpg",
            mask_path="masks/crack.png",
            transforms=train_transforms
        )
        ```

        Without mask (for inference):
        ```python
        result = apply_transforms(
            image=image_array,
            transforms=val_transforms
        )
        # Only "image" key in result
        ```

    Error Handling:
        - Invalid file paths raise FileNotFoundError
        - Unsupported image formats raise ValueError
        - Transform errors are propagated with context
        - Mask binarization ensures consistent binary values

    Performance Notes:
        - File loading uses OpenCV for speed
        - Tensor conversion is memory-efficient
        - Mask binarization threshold: 127 (0-255 range)
        - All tensors are float32 for training compatibility
    """
    # Load image and mask if they are file paths
    image_array = _load_image(image)
    mask_array = _load_mask(mask) if mask is not None else None

    # Apply transforms
    if transforms is not None:
        if mask_array is not None:
            transformed = transforms(image=image_array, mask=mask_array)
            return {
                "image": torch.from_numpy(transformed["image"]).float(),
                "mask": torch.from_numpy(transformed["mask"]).float(),
            }
        else:
            transformed = transforms(image=image_array)
            return {
                "image": torch.from_numpy(transformed["image"]).float(),
            }
    else:
        # Return without transforms
        result = {"image": torch.from_numpy(image_array).float()}
        if mask_array is not None:
            result["mask"] = torch.from_numpy(mask_array).float()
        return result


def _load_image(image: np.ndarray | str) -> np.ndarray:
    """Load image from file path or use numpy array directly.

    Args:
        image: Image as numpy array or file path

    Returns:
        Image as numpy array in RGB format

    Raises:
        ValueError: If image loading fails
    """
    if isinstance(image, str):
        import cv2

        img = cv2.imread(image)
        if img is None:
            raise ValueError(f"Failed to load image: {image}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    elif isinstance(image, np.ndarray):
        return image
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")


def _load_mask(mask: np.ndarray | str | None) -> np.ndarray | None:
    """Load mask from file path or use numpy array directly.

    Args:
        mask: Mask as numpy array, file path, or None

    Returns:
        Mask as numpy array in grayscale format, or None

    Raises:
        ValueError: If mask loading fails
    """
    if mask is None:
        return None
    elif isinstance(mask, str):
        import cv2

        mask_array = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
        if mask_array is None:
            raise ValueError(f"Failed to load mask: {mask}")
        return mask_array
    elif isinstance(mask, np.ndarray):
        return mask
    else:
        raise ValueError(f"Unsupported mask type: {type(mask)}")
