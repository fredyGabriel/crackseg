# ruff: noqa: PLR2004
import random
import typing
import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import PIL.Image
import PIL.ImageOps
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset

from src.data.validation import validate_data_config, validate_transform_config

# Import the transform functions
from .transforms import (
    apply_transforms,
    get_basic_transforms,
    get_transforms_from_config,
)

# Define SourceType at module level for type hinting
SourceType = str | Path | PIL.Image.Image | np.ndarray[Any, Any]
# Define cache_item_type at module level
CacheItemType = tuple[PIL.Image.Image | None, PIL.Image.Image | None]


class CrackSegmentationDataset(Dataset[Any]):
    """
    PyTorch Dataset for crack segmentation.
    Loads image/mask pairs from a provided list or scans directories.
    Applies transformations using Albumentations.
    """

    def __init__(  # noqa: PLR0913
        self,
        mode: str,  # Mode is required for transforms
        image_size: tuple[int, int] | None = None,
        # data_root: Optional[str] = None, # Keep for potential future use?
        samples_list: list[tuple[str, str]] | None = None,
        seed: int | None = None,
        in_memory_cache: bool = False,
        config_transform: dict[str, Any] | None = None,
        max_samples: int | None = None,  # New optional argument
    ):
        """
        Args:
            mode (str): 'train', 'val', or 'test'. Determines transforms.
            image_size (tuple[int, int] | None): Target size (height, width)
            for resizing.
            samples_list (list[tuple[str, str]] | None):
                Pre-defined list of (image_path, mask_path) tuples.
                If not provided, requires a different initialization
                method (e.g., via data_root, currently implies scanning,
                which is removed).
                If provided, these paths are used directly.
            seed (Optional[int]): Random seed for reproducibility.
            in_memory_cache (bool): If True, cache all data in RAM.
                Note: Cache stores raw PIL Images.
                Transforms applied after cache load.
            config_transform (dict[str, Any] | None): Dict with transform
            config (Hydra YAML).
            max_samples (Optional[int]): If set and > 0, limits the number
                of samples loaded for this dataset (for fast testing).
        """
        if mode not in ["train", "val", "test"]:
            msg = f"Invalid mode: {mode}. Use 'train', 'val', or 'test'."
            raise ValueError(msg)
        self.mode = mode
        self.seed = seed
        self.in_memory_cache = in_memory_cache
        # self.data_root = data_root # Store if needed later?
        self.samples: list[tuple[str, str]] = []

        if samples_list is not None:
            self.samples = samples_list
            if not self.samples:
                warnings.warn(
                    f"Provided samples_list for mode '{mode}' is empty.",
                    stacklevel=2,
                )
        # Removed the data_root scanning logic
        # elif data_root is not None:
        #     self._scan_directories() # Scan only if samples_list not given
        else:
            # If samples_list is None, we currently have no way to get data.
            # Raise error or expect data_root to be used by a subclass/factory?
            # For now, assume samples_list is the primary way.
            raise ValueError("samples_list must be provided.")

        # Limit the number of samples if max_samples is specified
        if max_samples is not None and max_samples > 0:
            original_count = len(self.samples)
            # Ensure we do not try to take more samples than available
            max_samples = min(max_samples, original_count)
            self.samples = self.samples[:max_samples]
            final_count = len(self.samples)
            print(
                f"DEBUG - Dataset '{mode}': Limited from "
                f"{original_count} to {final_count} samples"
            )
        else:
            print(
                f"DEBUG - Dataset '{mode}': Using all "
                f"{len(self.samples)} available samples (no limit)"
            )

        # Type hint for cache
        self._cache: list[tuple[Any, Any]] | None = None

        # Selection of transformations: config dict > image_size > default
        if config_transform is not None:
            self.transforms = get_transforms_from_config(
                config_transform, self.mode
            )
        elif image_size is not None:
            self.transforms = get_basic_transforms(
                mode=self.mode, image_size=image_size
            )
        else:
            # Fallback: use default values
            self.transforms = get_basic_transforms(mode=self.mode)

        # Build cache based on the final self.samples list
        if self.in_memory_cache and self.samples:
            self._build_cache()

        if self.seed is not None:
            self._set_seed()

    def _set_seed(self):
        """Set random seeds for reproducibility."""
        if self.seed is None:  # Guard clause
            return
        random.seed(self.seed)
        np.random.seed(self.seed)
        try:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
        except ImportError:
            warnings.warn(
                "torch is not installed. Cannot set torch seed.",
                stacklevel=2,
            )

    def _build_cache(self):
        """Cache all images and masks in memory as PIL Images."""
        self._cache = []
        for img_path, mask_path in self.samples:
            try:
                # Load as PIL for caching
                image = PIL.ImageOps.exif_transpose(
                    PIL.Image.open(img_path)
                ).convert("RGB")
                mask = PIL.ImageOps.exif_transpose(
                    PIL.Image.open(mask_path)
                ).convert("L")
                self._cache.append((image.copy(), mask.copy()))
                # Close files after copying
                image.close()
                mask.close()
            except (
                OSError,
                FileNotFoundError,
                PIL.UnidentifiedImageError,
                AttributeError,
                ValueError,
            ) as e:
                warnings.warn(
                    f"Could not cache image/mask: {img_path}, {mask_path}: "
                    f"{e}",
                    stacklevel=2,
                )
                # Placeholder for failed cache
                self._cache.append((None, None))

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Return the transformed image and mask pair at the given index.

        Loads image/mask from cache or disk.
        Applies the pre-defined transformation pipeline.
        Handles loading errors by attempting next sample.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing 'image' and 'mask'
                                     as torch tensors.

        Raises:
            RuntimeError: If no valid samples can be loaded after trying all.
        """
        attempts = 0
        max_attempts = len(self.samples)
        original_idx = idx

        while attempts < max_attempts:
            current_idx = (original_idx + attempts) % max_attempts
            # Define possible types for image/mask sources
            image_source: SourceType | None = None
            mask_source: SourceType | None = None

            # Try loading from cache first if enabled
            if self.in_memory_cache and self._cache is not None:
                cached_image, cached_mask = self._cache[current_idx]
                if cached_image is not None and cached_mask is not None:
                    # Convert PIL Image from cache to numpy array
                    image_source = np.array(cached_image)
                    mask_source = np.array(cached_mask)
                else:
                    # Cache entry failed or is missing, try next
                    attempts += 1
                    continue
            else:
                # Load paths from disk if not caching or cache failed
                image_source, mask_source = self.samples[current_idx]

            try:
                # Explicitly check if sources are paths (strings)
                if isinstance(image_source, str) and isinstance(
                    mask_source, str
                ):
                    # Load and transform from files
                    image = cv2.imread(image_source)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    mask = cv2.imread(mask_source, cv2.IMREAD_GRAYSCALE)

                    # Ensure mask is binary (0/1)
                    mask = (mask > 127).astype(np.uint8)

                    # Apply transformations
                    transformed = self.transforms(image=image, mask=mask)
                    image_tensor = transformed["image"]
                    mask_tensor = transformed["mask"]

                    # Ensure mask_tensor is binary (0/1)
                    if mask_tensor is None:
                        raise RuntimeError(
                            "Transform did not return a mask tensor."
                        )
                    mask_tensor = (mask_tensor > 0.5).float()
                    # Ensure shape (1, H, W)
                    if mask_tensor.ndim == 2:
                        mask_tensor = mask_tensor.unsqueeze(0)
                    return {"image": image_tensor, "mask": mask_tensor}
                else:
                    # Apply transformations using existing code (not strings)
                    sample = apply_transforms(
                        image=image_source,
                        mask=mask_source,
                        transforms=self.transforms,
                    )
                    mask_tensor = sample["mask"]
                    if mask_tensor.ndim == 2:
                        mask_tensor = mask_tensor.unsqueeze(0)
                    return {"image": sample["image"], "mask": mask_tensor}

            except (
                OSError,
                ValueError,
                TypeError,
                AttributeError,
                RuntimeError,
            ) as e:
                img_path, mask_path = self.samples[current_idx]
                # Adjust long warning message line
                warn_msg = (
                    f"Error processing sample at index {current_idx} "
                    f"({img_path}, {mask_path}): {e}. Skipping."
                )
                warnings.warn(warn_msg, stacklevel=2)
                attempts += 1
                # Optional: Invalidate cache entry?
                # if self.in_memory_cache and self._cache is not None:
                #     self._cache[current_idx] = (None, None)

        raise RuntimeError(
            "No valid image/mask pairs could be loaded/processed from dataset."
        )


def create_crackseg_dataset(  # noqa: PLR0913
    data_cfg: DictConfig,
    transform_cfg: DictConfig,
    mode: str,
    samples_list: list[tuple[str, str]],
    in_memory_cache: bool = False,
    max_samples: int | None = None,
) -> CrackSegmentationDataset:
    """
    Factory function to create a CrackSegmentationDataset from Hydra configs.

    Args:
        data_cfg (DictConfig): Data config (e.g. configs/data/default.yaml)
        transform_cfg (DictConfig): Transform config
            (e.g. configs/data/transform.yaml)
        mode (str): 'train', 'val' or 'test'
        samples_list (list[tuple[str, str]]): List of (image_path, mask_path)
            tuples
        in_memory_cache (bool): Whether to cache images in RAM
        max_samples (Optional[int]): Maximum number of samples for the dataset
    Returns:
        CrackSegmentationDataset: Configured dataset instance
    """
    # Improved debug message
    if max_samples is not None and max_samples > 0:
        print(f"DEBUG - Creating dataset for '{mode}' with:")
        print(f"  Total samples available: {len(samples_list)}")
        print(f"  Max samples limit: {max_samples}")
        print(
            f"  Will apply limit: {min(max_samples, len(samples_list))} "
            "samples"
        )
    else:
        print(
            f"DEBUG - Creating dataset for '{mode}' with all "
            f"{len(samples_list)} samples"
        )

    # Convert transform config to dict if needed
    transform_cfg_for_dataset: dict[Any, Any] | None = None
    container_result = OmegaConf.to_container(transform_cfg, resolve=True)
    if isinstance(container_result, dict):
        transform_cfg_for_dataset = typing.cast(
            dict[Any, Any], container_result
        )
    else:
        warn_msg = (
            f"OmegaConf.to_container did not return a dict for "
            f"transform_cfg. Got {type(container_result)}"
        )
        warnings.warn(warn_msg, stacklevel=2)
        # Optionally, raise an error or use a default value; for now, it
        # remains None

    # data_cfg is DictConfig by type hint, no isinstance needed
    seed_val = data_cfg.get("seed", 42)

    # Validate both configs
    # We assume that the validation functions can handle DictConfig or dict
    validate_data_config(data_cfg)
    validate_transform_config(transform_cfg)

    # Create the dataset
    dataset = CrackSegmentationDataset(
        mode=mode,
        samples_list=samples_list,
        seed=seed_val,  # Use the extracted seed
        in_memory_cache=in_memory_cache,
        config_transform=transform_cfg_for_dataset,  # Pass the dict or None
        max_samples=max_samples,
    )

    # Final report on dataset creation
    print(f"Created dataset for '{mode}' with {len(dataset)} samples.")
    return dataset
