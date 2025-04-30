"""Module providing functions for creating image transformation pipelines.

Uses albumentations to create transforms for different modes (train/val/test)
with support for resizing and normalization.
"""

import cv2
import numpy as np
import torch
from typing import Dict, Optional, Tuple, Union
from pathlib import Path

import albumentations as A
# Remove direct import
# from albumentations.pytorch import ToTensorV2


def get_basic_transforms(
    mode: str,
    image_size: Tuple[int, int] = (512, 512),
    # Restore ImageNet default values
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> A.Compose:
    """Creates a basic transformation pipeline based on the mode.

    Args:
        mode: Transformation mode ('train', 'val', 'test').
        image_size: Target image size (height, width).
        mean: Mean for per-channel normalization.
        std: Standard deviation for per-channel normalization.

    Returns:
        Albumentations transformation pipeline.

    Raises:
        ValueError: If the mode is invalid.
    """
    if mode not in ["train", "val", "test"]:
        msg = f"Invalid mode: {mode}. Must be 'train', 'val' or 'test'."
        raise ValueError(msg)

    # Core transforms
    core_transforms = [
        A.Resize(
            height=image_size[0],
            width=image_size[1],
            interpolation=cv2.INTER_LINEAR  # Default for image
        ),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        A.pytorch.ToTensorV2(),
    ]

    # Training augmentations
    image_augmentations = []
    if mode == "train":
        image_augmentations = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussNoise(p=0.2),
            # Use size parameter, remove height/width for v2.0.5 compat.
            A.RandomSizedCrop(
                min_max_height=(
                    int(0.8 * image_size[0]), int(1.2 * image_size[0])
                ),
                # 'size' expects original size, using target as approx.
                size=image_size,
                # height=image_size[0], # Invalid in 2.0.5
                # width=image_size[1],  # Invalid in 2.0.5
                p=1.0  # Ensure crop/resize always happens
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5
            ),
        ]
        # For train mode, exclude Resize from core transforms
        # as RandomSizedCrop handles resizing.
        final_transforms = image_augmentations + [
            t for t in core_transforms if not isinstance(t, A.Resize)
        ]
    else:
        # For val/test, use only core transforms (including Resize)
        final_transforms = core_transforms

    # Simple composition: Augmentations first, then core transforms.
    # Intensity/noise transforms should not affect mask by default.
    # Resize interpolation issues handled later in apply_transforms.
    # pipeline = A.Compose(image_augmentations + core_transforms)
    pipeline = A.Compose(final_transforms)

    return pipeline


def apply_transforms(
    image: Union[np.ndarray, str, Path],
    mask: Optional[Union[np.ndarray, str, Path]] = None,
    transforms: Optional[A.Compose] = None
) -> Dict[str, torch.Tensor]:
    """Applies transformations to an image and optionally its mask.

    Args:
        image: Numpy array or path to the image.
        mask: Numpy array or path to the mask (optional).
        transforms: Albumentations transformation pipeline.

    Returns:
        Dictionary with 'image' and optionally 'mask' keys.
    """
    # Load image if provided as path
    if isinstance(image, (str, Path)):
        image = cv2.imread(str(image))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load mask if provided as path
    if isinstance(mask, (str, Path)):
        mask = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)

    # Ensure mask is binary before transformations
    if mask is not None:
        # Use uint8 for masks before albumentations
        mask = (mask > 127).astype(np.uint8)

    # Apply transformations
    if transforms is not None:
        result = transforms(image=image, mask=mask)
        transformed_image = result["image"]
        if "mask" in result and result["mask"] is not None:
            transformed_mask = result["mask"]
            # Ensure mask remains binary and float afterwards
            # Convert potential interpolated values to int {0, 1} then to float
            transformed_mask = (transformed_mask > 0.5).long().float()
            return {"image": transformed_image, "mask": transformed_mask}
        # Return only image if no mask in result
        return {"image": transformed_image}

    # If no transforms, convert to tensor manually
    # Permute to (C, H, W) and normalize to [0, 1]
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    if mask is not None:
        # Convert mask to float tensor
        # Add channel dimension if necessary (H, W) -> (1, H, W)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        return {"image": image_tensor, "mask": mask_tensor}
    return {"image": image_tensor}


def get_transforms_from_config(config: dict, mode: str) -> A.Compose:
    """Create an Albumentations pipeline from a config dict and mode
    (train/val/test)."""
    # General settings
    resize_cfg = config.get('resize', {})
    normalize_cfg = config.get('normalize', {})

    transforms = []
    if resize_cfg.get('enabled', True):
        transforms.append(
            A.Resize(
                height=resize_cfg.get('height', 512),
                width=resize_cfg.get('width', 512),
                interpolation=cv2.INTER_LINEAR
            )
        )

    # Augmentations by mode
    aug_cfg = config.get(mode, {})
    if mode == 'train':
        if aug_cfg.get('random_crop', {}).get('enabled', False):
            crop = aug_cfg['random_crop']
            transforms.append(
                A.RandomCrop(
                    height=crop.get('height', 480),
                    width=crop.get('width', 480),
                    p=crop.get('p', 0.5)
                )
            )
        if aug_cfg.get('horizontal_flip', {}).get('enabled', False):
            transforms.append(
                A.HorizontalFlip(
                    p=aug_cfg['horizontal_flip'].get('p', 0.5)
                )
            )
        if aug_cfg.get('vertical_flip', {}).get('enabled', False):
            transforms.append(
                A.VerticalFlip(
                    p=aug_cfg['vertical_flip'].get('p', 0.5)
                )
            )
        if aug_cfg.get('rotate', {}).get('enabled', False):
            transforms.append(
                A.Rotate(
                    limit=aug_cfg['rotate'].get('limit', 90),
                    p=aug_cfg['rotate'].get('p', 0.5)
                )
            )
        if aug_cfg.get('color_jitter', {}).get('enabled', False):
            cj = aug_cfg['color_jitter']
            transforms.append(
                A.ColorJitter(
                    brightness=cj.get('brightness', 0.2),
                    contrast=cj.get('contrast', 0.2),
                    saturation=cj.get('saturation', 0.2),
                    hue=cj.get('hue', 0.1),
                    p=cj.get('p', 0.3)
                )
            )
    # For val/test, only minimal transforms (no augmentations)
    # (If needed, can add more logic here)

    # Normalization
    if normalize_cfg.get('enabled', True):
        transforms.append(
            A.Normalize(
                mean=normalize_cfg.get('mean', [0.485, 0.456, 0.406]),
                std=normalize_cfg.get('std', [0.229, 0.224, 0.225]),
                max_pixel_value=255.0
            )
        )
    transforms.append(
        A.pytorch.ToTensorV2()
    )
    return A.Compose(transforms)
