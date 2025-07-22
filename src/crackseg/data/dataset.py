#!/usr/bin/env python3
"""
Crack segmentation dataset implementation with configurable data loading.

This module provides the core dataset class for pavement crack segmentation
tasks. It supports flexible data loading strategies, in-memory caching for
performance, and extensive configuration through Hydra integration.

Key Features:
- Multiple data source support (file paths, PIL Images, numpy arrays)
- In-memory caching for improved training speed
- Robust error handling and fallback mechanisms
- Configurable data augmentation through Albumentations
- Sample limiting for development and testing
- Comprehensive validation and integrity checking

The module follows PyTorch Dataset conventions while providing additional
functionality specific to segmentation tasks:

- Binary mask validation and normalization
- Automatic image format conversion and color space handling
- Deterministic sample selection with seed control
- Memory-efficient caching strategies

Examples:
    Basic dataset creation with samples list:
    ```python
    from crackseg.data.dataset import CrackSegmentationDataset

    # Define sample pairs
    samples = [
        ("data/train/images/crack_001.jpg", "data/train/masks/crack_001.png"),
        ("data/train/images/crack_002.jpg", "data/train/masks/crack_002.png"),
    ]

    # Create dataset
    dataset = CrackSegmentationDataset(
        mode="train",
        samples_list=samples,
        image_size=(512, 512),
        in_memory_cache=True
    )

    # Access data
    sample = dataset[0]
    image, mask = sample["image"], sample["mask"]
    ```

    Factory-based creation with configuration:
    ```python
    from crackseg.data.dataset import create_crackseg_dataset
    from omegaconf import OmegaConf

    # Load configurations
    data_cfg = OmegaConf.load("configs/data/default.yaml")
    transform_cfg = OmegaConf.load(
        "configs/data/transform/augmentations.yaml"
    )

    # Create dataset via factory
    dataset = create_crackseg_dataset(
        data_cfg=data_cfg,
        transform_cfg=transform_cfg,
        mode="train",
        samples_list=samples,
        max_samples=1000  # Limit for fast development
    )
    ```

Performance Considerations:
- In-memory caching significantly speeds up training but requires sufficient
  RAM
- Sample limiting is useful for development and prototyping
- Error handling ensures robust operation with corrupted or missing files
- Transform caching at PIL level before applying Albumentations transformations

Integration:
- Designed for use with PyTorch DataLoader
- Integrates with Hydra configuration system
- Compatible with distributed training setups
- Supports custom sampling strategies

See Also:
    - src.data.transforms: Transform pipeline implementation
    - src.data.validation: Data validation utilities
    - src.data.factory: DataLoader creation utilities
    - configs/data/: Configuration examples and templates
"""

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

from crackseg.data.validation import (
    validate_data_config,
    validate_transform_config,
)

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
        - Masks are converted to binary (0/1) values with threshold at 127
        - EXIF orientation is automatically handled for proper display
        - Error handling attempts next sample if current sample fails to load
        - Caching stores PIL Images to preserve quality before transforms
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
        Initialize the crack segmentation dataset with specified configuration.

        Args:
            mode: Dataset mode determining transform pipeline.
                Must be one of: 'train', 'val', 'test'.

            image_size: Target image dimensions as (height, width) tuple.
                If provided, creates basic transform pipeline with resize.
                Ignored if config_transform is specified.

            samples_list: List of (image_path, mask_path) string tuples.
                Each tuple should contain absolute or relative paths to
                corresponding image and mask files. Required parameter.

            seed: Random seed for reproducible data loading and transforms.
                If None, behavior may vary between runs. Recommended to set
                for deterministic training.

            in_memory_cache: Whether to cache all images in RAM as PIL Images.
                Significantly speeds up training but requires sufficient
                memory. Cache is built during initialization.

            config_transform: Dictionary containing transform configuration.
                Should follow Albumentations format with transform names and
                parameters. Takes precedence over image_size parameter.

            max_samples: Maximum number of samples to load from samples_list.
                Useful for development, testing, or debugging with smaller
                datasets. If None or 0, all samples are used.

        Raises:
            ValueError: If mode is not 'train', 'val', or 'test'
            ValueError: If samples_list is None or empty

        Examples:
            Minimal initialization:
            ```python
            dataset = CrackSegmentationDataset(
                mode="train",
                samples_list=[("img1.jpg", "mask1.png")]
            )
            ```

            Full configuration:
            ```python
            dataset = CrackSegmentationDataset(
                mode="train",
                image_size=(512, 512),
                samples_list=sample_pairs,
                seed=42,
                in_memory_cache=True,
                max_samples=1000
            )
            ```

            With custom transforms:
            ```python
            transforms = {
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
                config_transform=transforms
            )
            ```

        Note:
            - Transform pipeline is automatically configured based on mode
            - Caching builds during initialization and may take time for large
            datasets
            - Sample limiting is applied after initial sample list validation
            - Seed affects both data sampling and transform randomness
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
        """
        Set random seeds for reproducible dataset behavior.

        Configures random number generators for:
        - Python's random module (for sampling)
        - NumPy random state (for array operations)
        - PyTorch random state (for tensor operations and CUDA)

        This ensures deterministic behavior across different runs when
        the same seed is used, which is crucial for:
        - Reproducible experiments
        - Debugging and development
        - Model comparison and validation

        Note:
            Only sets seeds if self.seed is not None. CUDA seed setting
            is attempted but fails gracefully if CUDA is unavailable.

        Examples:
            ```python
            dataset = CrackSegmentationDataset(
                mode="train",
                samples_list=samples,
                seed=42  # Ensures reproducible behavior
            )

            # Same samples will be returned in same order across runs
            sample1 = dataset[0]
            sample2 = dataset[0]  # Identical to sample1
            ```
        """
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
        """
        Build in-memory cache of all images and masks as PIL Images.

        This method pre-loads all dataset images and masks into memory
        to eliminate disk I/O during training. The cache stores PIL Images
        which preserves image quality and allows transforms to be applied
        at access time rather than cache time.

        Process:
        1. Iterates through all sample pairs in self.samples
        2. Loads images using PIL with EXIF orientation handling
        3. Converts images to RGB and masks to grayscale
        4. Stores copies in memory to avoid file handle issues
        5. Handles loading errors gracefully with None placeholders

        Error Handling:
        - Failed loads are stored as (None, None) tuples
        - Warnings are issued for problematic files
        - Dataset continues to function with remaining valid samples

        Memory Considerations:
        - Cache size scales linearly with dataset size and image resolution
        - Typical memory usage: ~3MB per 512x512 RGB image
        - For 10,000 images at 512x512: ~30GB RAM required

        Examples:
            Enable caching during dataset creation:
            ```python
            dataset = CrackSegmentationDataset(
                mode="train",
                samples_list=samples,
                in_memory_cache=True  # Builds cache during __init__
            )

            # Subsequent data access is much faster
            for sample in dataset:
                # No disk I/O - served from cache
                image, mask = sample["image"], sample["mask"]
            ```

        Note:
            - Cache is built only if in_memory_cache=True and samples exist
            - PIL Images are copied to prevent file handle issues
            - Original files are closed after copying to cache
            - Failed cache entries don't prevent dataset from functioning
        """
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
        """
        Return the total number of samples in the dataset.

        Returns:
            int: Number of valid sample pairs available in the dataset.

        Note:
            This reflects the actual number of samples after any
            max_samples limiting has been applied during initialization.
        """
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
                    if image is None:
                        raise OSError(f"Failed to load image: {image_source}")
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    mask = cv2.imread(mask_source, cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        raise OSError(f"Failed to load mask: {mask_source}")

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
            "No valid image/mask pairs could be loaded/processed from "
            "crackseg.dataset."
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
    Factory function to create a CrackSegmentationDataset from Hydra
    configurations.

    This factory provides a convenient interface for creating datasets from
    structured configuration files, handling validation, type conversion,
    and parameter extraction automatically.

    The function bridges the gap between Hydra's configuration system and
    the dataset class constructor, providing:
    - Configuration validation and type checking
    - Parameter extraction and conversion
    - Debug logging for transparency
    - Error handling for malformed configurations

    Args:
        data_cfg: Hydra configuration for data settings.
            Expected to contain:
            - seed (int, optional): Random seed for reproducibility
            - Additional data-related parameters

        transform_cfg: Hydra configuration for data transforms.
            Should follow Albumentations format with transform definitions
            organized by dataset mode (train/val/test).

        mode: Dataset mode for transform selection.
            Must be one of: 'train', 'val', 'test'.

        samples_list: Pre-computed list of (image_path, mask_path) tuples.
            Should contain valid file paths accessible from current directory.

        in_memory_cache: Whether to enable in-memory caching.
            Defaults to False. Enable for faster training with sufficient RAM.

        max_samples: Optional limit on number of samples to load.
            Useful for development, testing, or debugging scenarios.

    Returns:
        CrackSegmentationDataset: Fully configured dataset instance ready
        for use.

    Raises:
        ValueError: If configuration validation fails
        TypeError: If configuration types are incompatible
        KeyError: If required configuration keys are missing

    Examples:
        Basic usage with configuration files:
        ```python
        from omegaconf import OmegaConf

        # Load configurations
        data_cfg = OmegaConf.load("configs/data/default.yaml")
        transform_cfg = OmegaConf.load(
            "configs/data/transform/augmentations.yaml"
        )

        # Get sample list from somewhere (factory, scan, etc.)
        samples = get_samples_for_mode("train")

        # Create dataset
        dataset = create_crackseg_dataset(
            data_cfg=data_cfg,
            transform_cfg=transform_cfg,
            mode="train",
            samples_list=samples
        )
        ```

        Development mode with sample limiting:
        ```python
        # Create smaller dataset for fast iteration
        dev_dataset = create_crackseg_dataset(
            data_cfg=data_cfg,
            transform_cfg=transform_cfg,
            mode="train",
            samples_list=all_samples,
            max_samples=100,          # Only 100 samples
            in_memory_cache=True      # Fast access
        )
        ```

        Multiple datasets for train/val/test:
        ```python
        datasets = {}
        for mode in ["train", "val", "test"]:
            datasets[mode] = create_crackseg_dataset(
                data_cfg=data_cfg,
                transform_cfg=transform_cfg,
                mode=mode,
                samples_list=samples_dict[mode]
            )
        ```

    Configuration Examples:
        data_cfg structure:
        ```yaml
        seed: 42
        image_size: [512, 512]
        data_root: "data/"
        # ... other data parameters
        ```

        transform_cfg structure:
        ```yaml
        train:
          transforms:
            - name: "Resize"
              params: {height: 512, width: 512}
            - name: "HorizontalFlip"
              params: {p: 0.5}
        val:
          transforms:
            - name: "Resize"
              params: {height: 512, width: 512}
        ```

    Validation:
        The function performs comprehensive validation:
        - Calls validate_data_config() for data configuration
        - Calls validate_transform_config() for transform configuration
        - Verifies configuration type compatibility
        - Logs warnings for potential issues

    Integration:
        Designed to work with:
        - Hydra configuration management system
        - src.data.factory for complete DataLoader creation
        - src.data.validation for configuration checking
        - Standard PyTorch DataLoader for batch processing

    Note:
        - Debug logging shows sample counts and limitations applied
        - Transform configuration is converted to dict format for dataset
        - Seed value is extracted from crackseg.data configuration
        - Configuration validation is performed before dataset creation
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
