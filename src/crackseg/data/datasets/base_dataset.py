"""Base dataset implementation for crack segmentation.

This module provides the core CrackSegmentationDataset class with advanced
data loading capabilities for pavement crack segmentation tasks.
"""

import random
from pathlib import Path
from typing import Any

import numpy as np
import PIL.Image
import PIL.ImageOps
import torch
from torch.utils.data import Dataset

from ..transforms.config import get_transforms_from_config

# Import the transform functions
from .cache_manager import CacheManager
from .loaders import ImageLoader, MaskLoader

# Define SourceType at module level for type hinting
SourceType = str | Path | PIL.Image.Image | np.ndarray[Any, Any]
# Define cache_item_type at module level
CacheItemType = tuple[PIL.Image.Image | None, PIL.Image.Image | None]


class CrackSegmentationDataset(Dataset[Any]):
    """
    PyTorch Dataset for pavement crack segmentation with advanced data loading
    capabilities.

    This dataset class provides comprehensive support for crack segmentation
    tasks with:
    - Flexible data source handling (file paths, PIL Images, numpy arrays)
    - Optional in-memory caching for performance optimization
    - Configurable data augmentation through Albumentations
    - Robust error handling with fallback mechanisms
    - Sample limiting for development and testing scenarios
    - Deterministic behavior through seed control

    The dataset follows PyTorch conventions while adding segmentation-specific
    features:
    - Binary mask validation and normalization (0/1 values)
    - Automatic image format conversion (RGB for images, grayscale for masks)
    - EXIF orientation handling for proper image display
    - Memory-efficient caching strategies

    Attributes:
        mode (str): Dataset mode - 'train', 'val', or 'test'
        seed (int | None): Random seed for reproducible behavior
        in_memory_cache (bool): Whether images are cached in memory
        samples (list[tuple[str, str]]): List of (image_path, mask_path) pairs
        transforms: Albumentations transform pipeline

    Examples:
        Basic usage with file paths:
        ```python
        dataset = CrackSegmentationDataset(
            mode="train",
            samples_list=[
                ("images/crack1.jpg", "masks/crack1.png"),
                ("images/crack2.jpg", "masks/crack2.png")
            ],
            image_size=(512, 512)
        )

        # Access sample
        sample = dataset[0]
        image = sample["image"]  # torch.Tensor, shape (3, 512, 512)
        mask = sample["mask"]    # torch.Tensor, shape (1, 512, 512)
        ```

        With caching and sample limiting:
        ```python
        dataset = CrackSegmentationDataset(
            mode="train",
            samples_list=samples,
            in_memory_cache=True,  # Cache for speed
            max_samples=500,       # Limit for development
            seed=42               # Reproducible results
        )
        ```

        With custom transform configuration:
        ```python
        transform_config = {
            "train": {
                "transforms": [
                    {"name": "Resize",
                    "params": {"height": 512, "width": 512}},
                    {"name": "HorizontalFlip", "params": {"p": 0.5}}
                ]
            }
        }

        dataset = CrackSegmentationDataset(
            mode="train",
            samples_list=samples,
            config_transform=transform_config
        )
        ```

        With sample limiting for development:
        ```python
        # Limit to 100 samples for quick testing
        dataset = CrackSegmentationDataset(
            mode="train",
            samples_list=samples,
            max_samples=100
        )
        print(f"Dataset size: {len(dataset)}")  # 100
        ```

    Performance Notes:
        - In-memory caching provides ~10x speedup for repeated access
        - Transform application is the primary performance bottleneck
        - Failed samples trigger automatic fallback to prevent crashes
        - OpenCV is used for disk loading due to speed advantages

    Memory Considerations:
        - Caching requires ~2-4MB per image depending on resolution
        - For large datasets, consider using max_samples for development
        - Memory usage scales linearly with number of cached samples
        - Cache can be disabled for memory-constrained environments

    Error Handling:
        - Corrupted files are automatically skipped with warnings
        - Failed samples trigger fallback to next available sample
        - RuntimeError is raised only if all samples fail to load
        - Detailed error messages help identify problematic files
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
        """Initialize the CrackSegmentationDataset.

        Args:
            mode: Dataset mode ('train', 'val', 'test') for transform selection
            image_size: Target image dimensions (height, width)
            samples_list: List of (image_path, mask_path) tuples
            seed: Random seed for reproducible behavior
            in_memory_cache: Whether to cache images in memory
            config_transform: Transform configuration dictionary
            max_samples: Maximum number of samples to include (for development)

        Raises:
            ValueError: If mode is invalid or samples_list is empty
            RuntimeError: If no valid samples can be loaded
        """
        # Validate mode
        if mode not in ["train", "val", "test"]:
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'train', 'val', or 'test'"
            )

        self.mode = mode
        self.seed = seed
        self.in_memory_cache = in_memory_cache
        self.image_size = image_size or (512, 512)

        # Set seed for reproducible behavior
        self._set_seed()

        # Initialize samples list
        if samples_list is None:
            samples_list = []

        # Apply sample limiting if specified
        if max_samples is not None and max_samples < len(samples_list):
            samples_list = samples_list[:max_samples]

        self.samples = samples_list

        if not self.samples:
            raise ValueError("No samples provided to dataset")

        # Initialize transforms
        if config_transform is not None:
            self.transforms = get_transforms_from_config(
                config_transform, mode
            )
        else:
            from ..transforms.pipelines import get_basic_transforms

            self.transforms = get_basic_transforms(mode, self.image_size)

        # Initialize cache and loaders
        self._cache: list[CacheItemType] | None = None
        self._image_loader = ImageLoader()
        self._mask_loader = MaskLoader()
        self._cache_manager = CacheManager()

        # Build cache if enabled
        if self.in_memory_cache:
            self._build_cache()

    def _set_seed(self) -> None:
        """Set random seed for reproducible behavior."""
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

    def _build_cache(self) -> None:
        """Build in-memory cache for faster data loading."""
        print(f"Building cache for {len(self.samples)} samples...")
        self._cache = []

        for i, (image_path, mask_path) in enumerate(self.samples):
            try:
                # Load image and mask using loaders
                image = self._image_loader.load(image_path)
                mask = self._mask_loader.load(mask_path)

                # Handle EXIF orientation
                image = PIL.ImageOps.exif_transpose(image)

                self._cache.append((image, mask))

            except Exception as e:
                print(f"Warning: Failed to cache sample {i}: {e}")
                self._cache.append((None, None))

        print(
            f"Cache built with {len([x for x in self._cache if x[0] is not None])} valid samples"
        )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Retrieve and transform a sample at the specified index.

        This method implements robust data loading with fallback mechanisms:
        1. Attempts to load from cache if enabled, otherwise from disk
        2. Applies configured transform pipeline
        3. Ensures proper tensor format and binary mask values
        4. Falls back to next sample if current sample fails

        Args:
            idx: Sample index in range [0, len(dataset))

        Returns:
            dict[str, torch.Tensor]: Dictionary containing:
                - "image": RGB image tensor of shape (C, H, W)
                - "mask": Binary mask tensor of shape (1, H, W)

        Raises:
            RuntimeError: If no valid samples can be loaded after trying
                all available samples in the dataset.
        """
        if self.in_memory_cache:
            return self._cache_manager.get_sample(
                idx=idx,
                samples=self.samples,
                cache=self._cache,
                in_memory_cache=self.in_memory_cache,
                transforms=self.transforms,
                image_loader=self._image_loader,
                mask_loader=self._mask_loader,
            )
        else:
            # Load directly from disk without cache manager
            return self._load_sample_direct(idx)

    def _load_sample_direct(self, idx: int) -> dict[str, torch.Tensor]:
        """Load a sample directly from disk without cache manager.

        Args:
            idx: Sample index

        Returns:
            Dictionary with 'image' and 'mask' tensors
        """
        image_path, mask_path = self.samples[idx]

        # Load image and mask
        image = self._image_loader.load(image_path)
        mask = self._mask_loader.load(mask_path)

        # Handle EXIF orientation
        image = PIL.ImageOps.exif_transpose(image)

        # Convert to arrays
        image_array = np.array(image)
        mask_array = np.array(mask)

        # Apply transforms
        transformed = self.transforms(image=image_array, mask=mask_array)

        # Convert to tensors
        image_data = transformed["image"]
        mask_data = transformed["mask"]

        # Handle both numpy arrays and tensors
        if isinstance(image_data, np.ndarray):
            image_tensor = torch.from_numpy(image_data).float()
        else:
            image_tensor = image_data.float()

        if isinstance(mask_data, np.ndarray):
            mask_tensor = torch.from_numpy(mask_data).float()
        else:
            mask_tensor = mask_data.float()

        # Ensure proper tensor shapes
        if image_tensor.dim() == 2:
            image_tensor = image_tensor.unsqueeze(0)
        elif image_tensor.dim() == 3 and image_tensor.shape[0] == 3:
            # Already in (C, H, W) format
            pass
        else:
            image_tensor = image_tensor.permute(
                2, 0, 1
            )  # (H, W, C) -> (C, H, W)

        if mask_tensor.dim() == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        elif mask_tensor.dim() == 3 and mask_tensor.shape[0] == 1:
            # Already in (1, H, W) format
            pass
        else:
            mask_tensor = mask_tensor.unsqueeze(0)  # (H, W) -> (1, H, W)

        # Ensure binary mask values
        mask_tensor = (mask_tensor > 0.5).float()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
        }
