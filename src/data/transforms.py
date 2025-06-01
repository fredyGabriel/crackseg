"""Module providing functions for creating image transformation pipelines.

Uses albumentations to create transforms for different modes (train/val/test)
with support for resizing and normalization.
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
    core_transforms: Sequence[Any] = [
        A.Resize(
            height=image_size[0],
            width=image_size[1],
            interpolation=cv2.INTER_LINEAR,  # Default for image
        ),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        A.pytorch.ToTensorV2(),
    ]

    # Training augmentations
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
        final_transforms: Sequence[Any] = list(image_augmentations) + [
            t for t in core_transforms if not isinstance(t, A.Resize)
        ]
    else:
        final_transforms = core_transforms

    pipeline = A.Compose(list(final_transforms))  # type: ignore[arg-type]

    return pipeline


def _load_image(
    image: np.ndarray[Any, Any] | str | Path,
) -> np.ndarray[Any, Any]:
    if isinstance(image, str | Path):
        img_array = cv2.imread(str(image))
        return cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    return image


def _load_mask(
    mask: np.ndarray[Any, Any] | str | Path | None,
) -> np.ndarray[Any, Any] | None:
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
    """Applies transformations to an image and optionally its mask.

    Args:
        image: Numpy array or path to the image.
        mask: Numpy array or path to the mask (optional).
        transforms: Albumentations transformation pipeline.

    Returns:
        Dictionary with 'image' and optionally 'mask' keys. If no pipeline is
        provided, both image and mask are returned as torch.Tensor. The image
        will have shape (C, H, W) and the mask will have shape (1, H, W) if
        present, both normalized to [0, 1].

    Notes:
        - If no transforms are provided, the function will convert the image
          and mask to torch.Tensor, permuting the image to (C, H, W) and
          adding a channel dimension to the mask if needed.
        - The returned mask will always have shape (1, H, W) for consistency.
    """
    current_image: np.ndarray[Any, Any] = _load_image(image)
    current_mask: np.ndarray[Any, Any] | None = _load_mask(mask)

    if current_mask is not None:
        current_mask = (current_mask > MASK_BINARIZATION_THRESHOLD).astype(
            np.uint8
        )

    if transforms is not None:
        result = transforms(image=current_image, mask=current_mask)
        transformed_image = result["image"]
        if "mask" in result and result["mask"] is not None:
            transformed_mask = result["mask"]
            if isinstance(transformed_mask, np.ndarray):
                transformed_mask = torch.from_numpy(transformed_mask)
            transformed_mask = (
                (transformed_mask > MASK_THRESHOLD_FLOAT).long().float()
            )
            return {"image": transformed_image, "mask": transformed_mask}
        return {"image": transformed_image}

    image_tensor = (
        torch.from_numpy(current_image).permute(2, 0, 1).float() / 255.0
    )
    if current_mask is not None:
        mask_tensor = torch.from_numpy(current_mask)
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
    if isinstance(config_list, dict | DictConfig):
        specs_from_mapping: list[dict[str, Any] | DictConfig] = []
        for name_key, params_val in config_list.items():
            str_name_key = str(name_key)
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
    """Create an Albumentations pipeline from a list or dict of transform
    configs.

    Args:
        config_list (list[dict[str, Any]] | dict[str, Any]): Either:
            - List of dictionaries where each defines a transform
            ('name' and 'params')
            - Single dictionary with transform configs
            (e.g., {'resize': {...}})
        mode (str): 'train', 'val', or 'test'. Currently unused but kept for
            API consistency.

    Returns:
        A.Compose: The Albumentations pipeline.
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
            if name == "ToTensorV2":
                transform_class = ToTensorV2
                params_value = {}
            else:
                transform_class = getattr(A, name)

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
                pass

            transforms_pipeline.append(transform_class(**params_value))

        except AttributeError as exc:
            raise ValueError(f"Unknown transform name: '{name}'") from exc
        except Exception as e:
            raise ValueError(
                f"Error instantiating transform '{name}' with params "
                f"{params_value}: {e}"
            ) from e

    return A.Compose(transforms_pipeline)
