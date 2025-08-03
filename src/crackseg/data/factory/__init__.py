"""Factory patterns for data pipeline components.

This module provides factory functions for creating datasets and dataloaders
from configuration. It handles the complex orchestration of dataset creation,
data splitting, distributed training setup, and dataloader configuration.

Main exports:
    - create_dataloaders_from_config: Main factory function for complete data pipeline
    - create_dataset: Factory function for dataset creation
    - DataPipelineFactory: Class for data pipeline factory operations
    - DatasetFactory: Class for dataset factory operations
    - LoaderFactory: Class for dataloader factory operations
"""

from .dataset_factory import create_dataset
from .loader_factory import create_dataloaders_from_config
from .pipeline_factory import DataPipelineFactory

__all__ = [
    "create_dataloaders_from_config",
    "create_dataset",
    "DataPipelineFactory",
]
