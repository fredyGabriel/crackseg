"""Cache management for crack segmentation datasets.

This module provides efficient in-memory caching functionality for
crack segmentation datasets to improve data loading performance.
"""

import warnings

import numpy as np
import PIL.ImageOps
import torch
from albumentations import Compose

from .loaders import ImageLoader, MaskLoader
from .types import CacheItemType, SourceType


class CacheManager:
    """Manages in-memory caching for dataset samples.

    This class provides efficient caching mechanisms for crack segmentation
    datasets, handling both cached and disk-based loading with proper
    error handling and fallback mechanisms.

    Features:
        - Automatic cache hit/miss detection
        - Robust error handling with sample fallback
        - Memory-efficient tensor conversion
        - Binary mask validation and normalization
        - EXIF orientation handling
    """

    def get_sample(
        self,
        idx: int,
        samples: list[tuple[str, str]],
        cache: list[CacheItemType] | None,
        in_memory_cache: bool,
        transforms: Compose,
        image_loader: ImageLoader,
        mask_loader: MaskLoader,
    ) -> dict[str, torch.Tensor]:
        """Retrieve and transform a sample with caching support.

        Args:
            idx: Sample index
            samples: List of (image_path, mask_path) tuples
            cache: In-memory cache of loaded samples
            in_memory_cache: Whether caching is enabled
            transforms: Albumentations transform pipeline
            image_loader: Image loading utility
            mask_loader: Mask loading utility

        Returns:
            Dictionary with 'image' and 'mask' tensors

        Raises:
            RuntimeError: If no valid samples can be loaded
        """
        attempts = 0
        max_attempts = len(samples)
        original_idx = idx

        while attempts < max_attempts:
            current_idx = (original_idx + attempts) % max_attempts
            image_source: SourceType | None = None
            mask_source: SourceType | None = None

            # Try loading from cache first if enabled
            if in_memory_cache and cache is not None:
                cached_image, cached_mask = cache[current_idx]
                if cached_image is not None and cached_mask is not None:
                    # Check if cached items are already tensors
                    if isinstance(cached_image, torch.Tensor) and isinstance(
                        cached_mask, torch.Tensor
                    ):
                        # Use cached tensors directly
                        image_tensor = cached_image
                        mask_tensor = cached_mask

                        # Ensure proper tensor shapes
                        image_tensor = self._ensure_image_shape(image_tensor)
                        mask_tensor = self._ensure_mask_shape(mask_tensor)

                        # Ensure binary mask values
                        mask_tensor = (mask_tensor > 0.5).float()

                        return {
                            "image": image_tensor,
                            "mask": mask_tensor,
                        }
                    elif hasattr(cached_image, "convert") and hasattr(
                        cached_mask, "convert"
                    ):
                        # Cache contains PIL images, convert to arrays and apply transforms
                        image_array = np.array(cached_image)
                        mask_array = np.array(cached_mask)

                        # Apply transforms
                        transformed = self._apply_transforms(
                            image=image_array,
                            mask=mask_array,
                            transforms=transforms,
                        )

                        # Convert to tensors and ensure proper format
                        image_tensor = self._convert_to_tensor(
                            transformed["image"]
                        )
                        mask_tensor = self._convert_to_tensor(
                            transformed["mask"]
                        )

                        # Ensure proper tensor shapes
                        image_tensor = self._ensure_image_shape(image_tensor)
                        mask_tensor = self._ensure_mask_shape(mask_tensor)

                        # Ensure binary mask values
                        mask_tensor = (mask_tensor > 0.5).float()

                        return {
                            "image": image_tensor,
                            "mask": mask_tensor,
                        }
                    else:
                        # Cache contains file paths, load from disk
                        image_source = cached_image
                        mask_source = cached_mask
                else:
                    # Cache miss, load from disk
                    image_source = samples[current_idx][0]
                    mask_source = samples[current_idx][1]
            else:
                # Cache disabled, load directly from disk
                image_source = samples[current_idx][0]
                mask_source = samples[current_idx][1]

            try:
                # Load image and mask
                image = image_loader.load(image_source)
                mask = mask_loader.load(mask_source)

                # Handle EXIF orientation for images
                image = PIL.ImageOps.exif_transpose(image)

                # Convert PIL images to numpy arrays for apply_transforms
                image_array = np.array(image)
                mask_array = np.array(mask)

                # Apply transforms
                transformed = self._apply_transforms(
                    image=image_array,
                    mask=mask_array,
                    transforms=transforms,
                )

                # Convert to tensors and ensure proper format
                image_tensor = self._convert_to_tensor(transformed["image"])
                mask_tensor = self._convert_to_tensor(transformed["mask"])

                # Ensure proper tensor shapes
                image_tensor = self._ensure_image_shape(image_tensor)
                mask_tensor = self._ensure_mask_shape(mask_tensor)

                # Ensure binary mask values
                mask_tensor = (mask_tensor > 0.5).float()

                return {
                    "image": image_tensor,
                    "mask": mask_tensor,
                }

            except Exception as e:
                # Log warning and try next sample
                warnings.warn(
                    f"Failed to load sample {current_idx}: {e}. "
                    f"Trying next sample...",
                    stacklevel=2,
                )
                attempts += 1

        # If we get here, all samples failed
        raise RuntimeError(
            f"Failed to load any valid samples after trying all "
            f"{max_attempts} samples in the dataset."
        )

    def _apply_transforms(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        transforms: Compose,
    ) -> dict[str, np.ndarray]:
        """Apply Albumentations transforms to image and mask.

        Args:
            image: Input image as numpy array
            mask: Input mask as numpy array
            transforms: Albumentations transform pipeline

        Returns:
            Dictionary with transformed image and mask
        """
        return transforms(image=image, mask=mask)

    def _convert_to_tensor(self, array: np.ndarray) -> torch.Tensor:
        """Convert numpy array to PyTorch tensor.

        Args:
            array: Input numpy array

        Returns:
            PyTorch tensor with float32 dtype
        """
        return torch.from_numpy(array).float()

    def _ensure_image_shape(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure image tensor has proper shape (C, H, W).

        Args:
            tensor: Input image tensor

        Returns:
            Tensor with proper shape (C, H, W)
        """
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() == 3:
            # Ensure channels are in the first dimension
            if tensor.shape[2] == 3:  # (H, W, C)
                tensor = tensor.permute(2, 0, 1)  # (C, H, W)

        return tensor

    def _ensure_mask_shape(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure mask tensor has proper shape (1, H, W).

        Args:
            tensor: Input mask tensor

        Returns:
            Tensor with proper shape (1, H, W)
        """
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)  # (1, H, W)

        return tensor
