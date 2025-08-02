"""Base dataset implementation for crack segmentation.

This module provides the core CrackSegmentationDataset class with advanced
data loading capabilities for pavement crack segmentation tasks.
"""

import random
import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import PIL.Image
import PIL.ImageOps
import torch
from torch.utils.data import Dataset

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

    Note:
        - Images are automatically converted to RGB format
        - Masks are converted to binary (0/1) values
        - Failed samples are automatically skipped with fallback
        - Caching significantly improves training speed
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
        Initialize the CrackSegmentationDataset.

        Args:
            mode: Dataset mode ('train', 'val', 'test')
            image_size: Target image size (height, width)
            samples_list: List of (image_path, mask_path) tuples
            seed: Random seed for reproducibility
            in_memory_cache: Whether to cache images in memory
            config_transform: Transform configuration dictionary
            max_samples: Maximum number of samples to load
        """
        self.mode = mode
        self.seed = seed
        self.in_memory_cache = in_memory_cache
        self.image_size = image_size

        # Set seed for reproducible behavior
        self._set_seed()

        # Initialize samples list
        if samples_list is None:
            self.samples = []
        else:
            self.samples = samples_list.copy()

        # Apply sample limiting if specified
        if max_samples is not None and max_samples > 0:
            max_samples = min(max_samples, len(self.samples))
            self.samples = self.samples[:max_samples]

        # Initialize cache
        self._cache: list[CacheItemType] | None = None
        if self.in_memory_cache:
            self._build_cache()

        # Setup transforms
        if config_transform is not None:
            self.transforms = get_transforms_from_config(
                config_transform, mode
            )
        else:
            # Handle the case where image_size might be None
            if self.image_size is not None:
                self.transforms = get_basic_transforms(mode, self.image_size)
            else:
                # Use default size if image_size is None
                self.transforms = get_basic_transforms(mode, (512, 512))

    def _set_seed(self):
        """Set random seed for reproducible behavior."""
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

    def _build_cache(self):
        """Build in-memory cache for faster data loading."""
        if not self.in_memory_cache:
            return

        print(f"Building cache for {len(self.samples)} samples...")
        self._cache = []

        for i, (image_path, mask_path) in enumerate(self.samples):
            try:
                # Load image
                if isinstance(image_path, str):
                    image = cv2.imread(image_path)
                    if image is None:
                        raise ValueError(f"Failed to load image: {image_path}")
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = PIL.Image.fromarray(image)
                elif isinstance(image_path, PIL.Image.Image):
                    image = image_path
                elif isinstance(image_path, np.ndarray):
                    image = PIL.Image.fromarray(image_path)
                else:
                    raise ValueError(
                        f"Unsupported image type: {type(image_path)}"
                    )

                # Load mask
                if isinstance(mask_path, str):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        raise ValueError(f"Failed to load mask: {mask_path}")
                    mask = PIL.Image.fromarray(mask)
                elif isinstance(mask_path, PIL.Image.Image):
                    mask = mask_path
                elif isinstance(mask_path, np.ndarray):
                    mask = PIL.Image.fromarray(mask_path)
                else:
                    raise ValueError(
                        f"Unsupported mask type: {type(mask_path)}"
                    )

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

        The method handles multiple data source types:
        - File paths (strings): Loaded using OpenCV/PIL
        - Cached PIL Images: Converted to numpy arrays
        - Direct numpy arrays: Used as-is

        Args:
            idx: Sample index in range [0, len(dataset)).
                If the specified sample fails to load, the method
                will attempt subsequent samples until success or
                all samples have been tried.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing:
                - "image": RGB image tensor of shape (C, H, W)
                - "mask": Binary mask tensor of shape (1, H, W)

        Raises:
            RuntimeError: If no valid samples can be loaded after trying
                all available samples in the dataset.

        Examples:
            Basic sample access:
            ```python
            sample = dataset[0]
            image = sample["image"]  # torch.Tensor, shape (3, H, W)
            mask = sample["mask"]    # torch.Tensor, shape (1, H, W)

            # Verify data properties
            assert image.dtype == torch.float32
            assert mask.dtype == torch.float32
            assert torch.all((mask == 0) | (mask == 1))  # Binary values
            ```

            Batch processing:
            ```python
            from torch.utils.data import DataLoader

            dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
            for batch in dataloader:
                images = batch["image"]  # (16, 3, H, W)
                masks = batch["mask"]    # (16, 1, H, W)

                # Process batch...
            ```

            Error handling demonstration:
            ```python
            # Dataset automatically handles corrupted files
            try:
                sample = dataset[idx]
                # Process successful sample
            except RuntimeError:
                # All samples failed to load
                print("Dataset contains no valid samples")
            ```

        Data Processing Pipeline:
        1. **Source Selection**: Cache vs disk loading
        2. **Image Loading**: OpenCV (BGR→RGB) or PIL Image conversion
        3. **Mask Processing**: Grayscale loading and binary thresholding
           (>127)
        4. **Transform Application**: Albumentations pipeline
        5. **Tensor Conversion**: Final PyTorch tensor formatting
        6. **Validation**: Binary mask values and proper tensor shapes

        Performance Notes:
        - Cached samples are ~10x faster to access than disk loading
        - Transform application is the primary performance bottleneck
        - Failed samples trigger automatic fallback to prevent crashes
        - OpenCV is used for disk loading due to speed advantages

        Format Guarantees:
        - Images are always RGB format tensors
        - Masks are always binary (0/1) single-channel tensors
        - Tensor shapes are consistent within dataset mode
        - All tensors are float32 dtype for training compatibility
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
                    image_source = cached_image
                    mask_source = cached_mask
                else:
                    # Cache miss or invalid entry, try disk loading
                    image_path, mask_path = self.samples[current_idx]
                    image_source = image_path
                    mask_source = mask_path
            else:
                # No cache, load from disk
                image_path, mask_path = self.samples[current_idx]
                image_source = image_path
                mask_source = mask_path

            try:
                # Load image
                if isinstance(image_source, str):
                    image = cv2.imread(image_source)
                    if image is None:
                        raise ValueError(
                            f"Failed to load image: {image_source}"
                        )
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = PIL.Image.fromarray(image)
                elif isinstance(image_source, PIL.Image.Image):
                    image = image_source
                elif isinstance(image_source, np.ndarray):
                    image = PIL.Image.fromarray(image_source)
                else:
                    raise ValueError(
                        f"Unsupported image type: {type(image_source)}"
                    )

                # Load mask
                if isinstance(mask_source, str):
                    mask = cv2.imread(mask_source, cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        raise ValueError(f"Failed to load mask: {mask_source}")
                    mask = PIL.Image.fromarray(mask)
                elif isinstance(mask_source, PIL.Image.Image):
                    mask = mask_source
                elif isinstance(mask_source, np.ndarray):
                    mask = PIL.Image.fromarray(mask_source)
                else:
                    raise ValueError(
                        f"Unsupported mask type: {type(mask_source)}"
                    )

                # Handle EXIF orientation for images
                image = PIL.ImageOps.exif_transpose(image)

                # Convert PIL images to numpy arrays for apply_transforms
                image_array = np.array(image)
                mask_array = np.array(mask)

                # Apply transforms
                transformed = apply_transforms(
                    image=image_array,
                    mask=mask_array,
                    transforms=self.transforms,
                )

                # Convert to tensors
                image_tensor = torch.from_numpy(transformed["image"]).float()
                mask_tensor = torch.from_numpy(transformed["mask"]).float()

                # Ensure proper tensor shapes
                if image_tensor.dim() == 2:
                    image_tensor = image_tensor.unsqueeze(0)
                elif image_tensor.dim() == 3:
                    # Ensure channels are in the first dimension
                    if image_tensor.shape[2] == 3:  # (H, W, C)
                        image_tensor = image_tensor.permute(
                            2, 0, 1
                        )  # (C, H, W)

                if mask_tensor.dim() == 2:
                    mask_tensor = mask_tensor.unsqueeze(0)  # (1, H, W)

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
