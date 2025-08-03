"""Dataset creation utilities for data loader factory.

This module provides utilities for creating and managing datasets with
optimized splitting and configuration handling.
"""

from typing import Any

from omegaconf import DictConfig
from torch.utils.data import Dataset

from crackseg.data.dataset import CrackSegmentationDataset
from crackseg.data.utils.splitting import (
    DatasetCreationConfig,
    create_split_datasets,
)


def _create_or_load_split_datasets(
    data_config: DictConfig,
    transform_config: DictConfig,
    dataloader_config: DictConfig,
    dataset_class: type[CrackSegmentationDataset],
) -> dict[str, Dataset[Any]]:
    """Create or load split datasets with optimized splitting.

    Args:
        data_config: Data configuration.
        transform_config: Transform configuration.
        dataloader_config: DataLoader configuration.
        dataset_class: Dataset class to use.

    Returns:
        Dictionary mapping split names to datasets.
    """
    # Create split datasets
    config = DatasetCreationConfig(
        data_root=data_config.data_root,
        transform_cfg=transform_config,
        dataset_cls=dataset_class,
    )

    split_datasets = create_split_datasets(config)

    return split_datasets


def create_dataset_pipeline(
    data_config: DictConfig,
    transform_config: DictConfig,
    dataset_class: type[CrackSegmentationDataset] = CrackSegmentationDataset,
) -> dict[str, Dataset[Any]]:
    """Create complete dataset pipeline from configuration.

    Args:
        data_config: Data configuration.
        transform_config: Transform configuration.
        dataset_class: Dataset class to use.

    Returns:
        Dictionary mapping split names to datasets.
    """
    return _create_or_load_split_datasets(
        data_config=data_config,
        transform_config=transform_config,
        dataloader_config=data_config,  # Use data_config as fallback
        dataset_class=dataset_class,
    )
