"""Dataset factory for creating CrackSegmentationDataset instances.

This module provides factory functions for creating datasets from Hydra
configurations with proper validation and parameter extraction.
"""

import typing
import warnings
from typing import Any

from omegaconf import DictConfig, OmegaConf

from crackseg.data.validation import (
    validate_data_config,
    validate_transform_config,
)

from .base_dataset import CrackSegmentationDataset


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

    Configuration Format:
        The function expects configurations in the following format:

        data_cfg:
        ```yaml
        seed: 42
        image_size: [512, 512]
        ```

        transform_cfg:
        ```yaml
        train:
          augmentations:
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
    elif isinstance(container_result, list):
        # Handle list-based transform configurations
        # This is normal for split-specific transforms like train/val/test
        transform_cfg_for_dataset = None
    else:
        warn_msg = (
            f"OmegaConf.to_container returned unexpected type for "
            f"transform_cfg. Got {type(container_result)}, expected dict or "
            "list"
        )
        warnings.warn(warn_msg, stacklevel=2)
        # Optionally, raise an error or use a default value; for now, it
        # remains None

    # Process transform configuration for dataset
    if transform_cfg_for_dataset is not None:
        # Handle different transform configuration formats
        # Case 1: transform_cfg is the complete config with all splits
        # Case 2: transform_cfg is already the specific split config
        if mode in transform_cfg_for_dataset:
            # Case 1: Complete config, extract the specific mode
            if "augmentations" in transform_cfg_for_dataset[mode]:
                transform_cfg_for_dataset = transform_cfg_for_dataset[mode][
                    "augmentations"
                ]
            else:
                transform_cfg_for_dataset = transform_cfg_for_dataset[mode]
        else:
            # Case 2: Already the specific split config
            if "augmentations" in transform_cfg_for_dataset:
                transform_cfg_for_dataset = transform_cfg_for_dataset[
                    "augmentations"
                ]

    # data_cfg is DictConfig by type hint, no isinstance needed
    seed_val = data_cfg.get("seed", 42)

    # Validate both configs
    # We assume that the validation functions can handle DictConfig or dict
    validate_data_config(data_cfg)

    # Handle different transform configuration formats
    # Case 1: transform_cfg is the complete config with all splits
    # Case 2: transform_cfg is already the specific split config
    if mode in transform_cfg:
        # Case 1: Complete config, extract the specific mode
        if "augmentations" in transform_cfg[mode]:
            validate_transform_config(transform_cfg[mode].augmentations)
        else:
            # Fallback: try to validate the mode directly
            validate_transform_config(transform_cfg[mode])
    else:
        # Case 2: Already the specific split config
        if "augmentations" in transform_cfg:
            validate_transform_config(transform_cfg.augmentations)
        else:
            # Fallback: try to validate directly
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
