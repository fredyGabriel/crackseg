"""Module providing functions for creating image transformation pipelines.

Uses albumentations to create transforms for different modes (train/val/test)
with support for resizing and normalization.
"""

import cv2
import numpy as np
import torch
from typing import Dict, Optional, Tuple, Union
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

import albumentations as A
from albumentations.pytorch import ToTensorV2


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


def get_transforms_from_config(config_list: Union[list, dict], mode: str
                               ) -> A.Compose:
    """Create an Albumentations pipeline from a list or dict of transform
    configs.

    Args:
        config_list (Union[list, dict]): Either:
            - List of dictionaries where each defines a transform
            ('name' and 'params')
            - Single dictionary with transform configs
            (e.g., {'resize': {...}})
        mode (str): 'train', 'val', or 'test'. Currently unused but kept for
            API consistency.

    Returns:
        A.Compose: The Albumentations pipeline.
    """
    transforms = []

    # Convert dict to list of transform specs if needed
    if isinstance(config_list, (dict, DictConfig)):
        # Create a list of transforms from a dictionary
        transform_specs = []
        for transform_name, params in config_list.items():
            if transform_name.lower() == 'resize':
                # Special handling for resize which takes height/width
                transform_specs.append({
                    "name": "Resize",
                    "params": params
                })
            else:
                # For other transforms
                transform_specs.append({
                    "name": transform_name,
                    "params": params
                })
        config_list = transform_specs

    # Process each transform defined in the list
    for transform_item in config_list:
        if not isinstance(transform_item, (dict, DictConfig)):
            raise ValueError("Each item in config_list must be a dictionary.")

        name = transform_item.get("name")
        params = transform_item.get("params", {})

        if name is None:
            raise ValueError("Each transform item must have a 'name' key.")

        try:
            # Find the transform class in Albumentations
            if name == "ToTensorV2":  # Handle specific case
                transform_class = ToTensorV2
                params = {}  # ToTensorV2 takes no params
            else:
                transform_class = getattr(A, name)

            # Instantiate and add to the list
            # Convert OmegaConf DictConfig params if necessary
            if isinstance(params, DictConfig):
                params = OmegaConf.to_container(params, resolve=True)

            # Standard instantiation works now as YAML provides correct params
            transforms.append(transform_class(**params))

        except AttributeError:
            raise ValueError(f"Unknown transform name: '{name}'")
        except Exception as e:
            raise ValueError(
                f"Error instantiating transform '{name}' with params {params}:"
                f"{e}"
            ) from e

    # Create the composition
    return A.Compose(transforms)
