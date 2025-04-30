import os
import warnings
import torch
from torch.utils.data import DataLoader, Dataset

from .dataset import CrackSegmentationDataset
from omegaconf import DictConfig, OmegaConf


def validate_data_config(data_cfg):
    """
    Validates the dataset configuration dictionary.
    Raises ValueError if required parameters are missing or invalid.
    """
    required_keys = [
        "data_root", "train_split", "val_split", "test_split", "image_size"
    ]
    for key in required_keys:
        if key not in data_cfg:
            raise ValueError(f"Missing required data config key: '{key}'")
    # Check split ratios sum to 1.0 (allowing small float error)
    total = (
        float(data_cfg["train_split"]) +
        float(data_cfg["val_split"]) +
        float(data_cfg["test_split"])
    )
    if not abs(total - 1.0) < 1e-4:
        raise ValueError(f"train/val/test splits must sum to 1.0, got {total}")
    # Check image_size is a list/tuple of length 2
    img_size = data_cfg["image_size"]
    if not (isinstance(img_size, (list, tuple)) and len(img_size) == 2):
        raise ValueError("image_size must be a list or tuple of length 2")


def validate_transform_config(transform_cfg):
    """
    Validates the transform configuration dictionary.
    Raises ValueError if required parameters are missing or invalid.
    """
    # General settings
    if "resize" not in transform_cfg:
        raise ValueError("Missing 'resize' section in transform config")
    resize = transform_cfg["resize"]
    for k in ["height", "width"]:
        if k not in resize:
            raise ValueError(f"Missing '{k}' in 'resize' config")
    # Normalization
    if "normalize" not in transform_cfg:
        raise ValueError("Missing 'normalize' section in transform config")
    norm = transform_cfg["normalize"]
    for k in ["mean", "std"]:
        if k not in norm:
            raise ValueError(f"Missing '{k}' in 'normalize' config")
    # Check mean/std are lists of length 3
    if not (
        isinstance(norm["mean"], (list, tuple)) and len(norm["mean"]) == 3
    ):
        raise ValueError("normalize.mean must be a list of 3 values")
    if not (
        isinstance(norm["std"], (list, tuple)) and len(norm["std"]) == 3
    ):
        raise ValueError("normalize.std must be a list of 3 values")


def create_crackseg_dataset(
    data_cfg: DictConfig,
    transform_cfg: DictConfig,
    mode: str,
    samples_list: list,
    in_memory_cache: bool = False
) -> CrackSegmentationDataset:
    """
    Factory function to create a CrackSegmentationDataset from Hydra configs.

    Args:
        data_cfg (DictConfig): Data config (e.g. configs/data/default.yaml)
        transform_cfg (DictConfig): Transform config
            (e.g. configs/data/transform.yaml)
        mode (str): 'train', 'val' or 'test'
        samples_list (list): List of (image_path, mask_path) tuples
        in_memory_cache (bool): Whether to cache images in RAM
    Returns:
        CrackSegmentationDataset: Configured dataset instance
    """
    # Convert transform config to dict if needed
    if isinstance(transform_cfg, DictConfig):
        transform_cfg = OmegaConf.to_container(transform_cfg, resolve=True)
    if isinstance(data_cfg, DictConfig):
        data_cfg = OmegaConf.to_container(data_cfg, resolve=True)
    # Validar ambos configs
    validate_data_config(data_cfg)
    validate_transform_config(transform_cfg)
    seed = data_cfg.get('seed', 42)
    return CrackSegmentationDataset(
        mode=mode,
        samples_list=samples_list,
        seed=seed,
        in_memory_cache=in_memory_cache,
        config_transform=transform_cfg
    )


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = -1,  # Default to auto-detect
    shuffle: bool = True,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    **kwargs
) -> DataLoader:
    """
    Creates and configures a PyTorch DataLoader with sensible defaults.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int): How many samples per batch to load. Default: 32.
        num_workers (int): How many subprocesses to use for data loading.
                           -1 attempts to use os.cpu_count() // 2. 0 means
                           data will be loaded in the main process.
                           Default: -1.
        shuffle (bool): Set to True to have the data reshuffled at every epoch.
                        Default: True (common for training).
        pin_memory (bool): If True, the data loader will copy Tensors into CUDA
                           pinned memory before returning them. Recommended
                           for GPU training. Default: True.
        prefetch_factor (int): Number of batches loaded in advance by each
                               worker. Default: 2.
        **kwargs: Additional keyword arguments to pass to the DataLoader
                  constructor.

    Returns:
        DataLoader: A configured PyTorch DataLoader instance.

    Raises:
        ValueError: If batch_size or prefetch_factor are not positive,
                    or if num_workers is less than -1.
    """
    # --- Parameter Validation ---
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if prefetch_factor <= 0:
        raise ValueError(
            f"prefetch_factor must be positive, got {prefetch_factor}"
        )
    if num_workers < -1:
        raise ValueError(f"num_workers must be >= -1, got {num_workers}")

    # --- Determine num_workers ---
    actual_num_workers = 0
    if num_workers == -1:
        try:
            cpu_count = os.cpu_count()
            if cpu_count is not None:
                actual_num_workers = max(1, cpu_count // 2)
            else:
                warnings.warn(
                    "Could not determine CPU count, defaulting num_workers to \
1."
                )
                actual_num_workers = 1
        except NotImplementedError:
            warnings.warn(
                "os.cpu_count() not implemented, defaulting num_workers to 1."
            )
            actual_num_workers = 1
    else:
        actual_num_workers = num_workers

    # --- Determine pin_memory ---
    # pin_memory only works on CUDA devices
    can_pin_memory = pin_memory and torch.cuda.is_available()
    if pin_memory and not can_pin_memory:
        warnings.warn(
            "pin_memory=True requires CUDA availability. "
            "Setting pin_memory=False."
        )

    # --- Create DataLoader ---
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=actual_num_workers,
        pin_memory=can_pin_memory,
        # prefetch needs workers
        prefetch_factor=prefetch_factor if actual_num_workers > 0 else None,
        # Keep workers alive
        persistent_workers=True if actual_num_workers > 0 else False,
        **kwargs
    )

    return dataloader
