"""Data loading for training pipeline.

This module provides data loading functions for the training pipeline
including dataloader creation, configuration validation, and error handling.
"""

import logging
import os
from typing import Any

import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from crackseg.data.factory import create_dataloaders_from_config
from crackseg.utils import DataError

# Configure standard logger
log = logging.getLogger(__name__)


def load_data(cfg: DictConfig) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """
    Load and create training and validation data loaders from configuration.

    This function handles the complete data loading pipeline including:
    - Configuration validation and path resolution
    - Transform pipeline creation
    - DataLoader instantiation with optimized settings
    - Error handling for missing or invalid data

    Args:
        cfg: Hydra configuration containing data settings.
            Expected structure:
            - cfg.data.data_root (str): Root directory for dataset
            - cfg.data.transform (DictConfig, optional): Transform
                configuration
            - cfg.data.dataloader (DictConfig, optional): DataLoader
                configuration

    Returns: tuple[DataLoader[Any], DataLoader[Any]]: A tuple containing:
            - train_loader: DataLoader for training data
            - val_loader: DataLoader for validation data

    Raises:
        DataError: If data loading fails due to:
            - Missing or invalid dataset files
            - Configuration errors
            - DataLoader creation failures
        OSError: If data directory is not accessible
        FileNotFoundError: If required data files are missing

    Examples:
        ```python
        cfg = OmegaConf.load("configs/data/default.yaml")
        train_loader, val_loader = load_data(cfg)

        # Check data loader properties
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")

        # Sample a batch
        images, masks = next(iter(train_loader))
        print(f"Batch shape: {images.shape}, {masks.shape}")
        ```

    Note:
        The function automatically resolves relative paths using Hydra's
        original working directory and provides fallback configurations
        for missing transform or dataloader settings.
    """
    log.info("Loading data...")
    try:
        # Handle experiment namespace configuration access
        data_cfg = None

        # Look for experiment-namespaced data configuration
        if "experiments" in cfg:
            # Look in experiments namespace for data config
            for exp_name in cfg.experiments:
                exp_config = cfg.experiments[exp_name]
                if hasattr(exp_config, "data"):
                    data_cfg = exp_config.data
                    break

        # Fall back to direct access
        if data_cfg is None and hasattr(cfg, "data"):
            data_cfg = cfg.data

        if data_cfg is None:
            raise AttributeError("No data configuration found in config")
        transform_cfg = None
        if hasattr(data_cfg, "transform"):
            transform_cfg = data_cfg.transform
        elif "data/transform" in cfg:
            transform_cfg = cfg["data/transform"]
        else:
            log.warning(
                "Transform config not found in Hydra config. "
                "Using empty config."
            )
            from omegaconf import OmegaConf

            transform_cfg = OmegaConf.create({})

        # Convert data_cfg to mutable dict for path resolution
        from omegaconf import OmegaConf

        data_cfg_dict = OmegaConf.to_container(data_cfg, resolve=True)
        if not isinstance(data_cfg_dict, dict):
            raise ValueError(
                f"Expected dict from data config, got {type(data_cfg_dict)}"
            )

        orig_cwd = hydra.utils.get_original_cwd()
        data_root = os.path.join(
            orig_cwd, data_cfg_dict.get("data_root", "data/")
        )
        data_cfg_dict["data_root"] = data_root

        # Convert back to DictConfig for consistency
        data_cfg = OmegaConf.create(data_cfg_dict)

        dataloader_cfg = None
        if hasattr(data_cfg, "dataloader"):
            dataloader_cfg = data_cfg.dataloader
        elif "data/dataloader" in cfg:
            dataloader_cfg = cfg["data/dataloader"]
        else:
            log.warning(
                "Dataloader config not found in Hydra config. "
                "Using data config as fallback."
            )
            dataloader_cfg = data_cfg

        # Ensure dataloader_cfg is a DictConfig
        if not isinstance(dataloader_cfg, DictConfig):
            # Attempt to convert if it's a basic dict or list that OmegaConf
            # can handle
            try:
                converted_cfg = OmegaConf.create(dataloader_cfg)
                if isinstance(converted_cfg, DictConfig):
                    dataloader_cfg = converted_cfg
                else:
                    log.warning(
                        "Could not convert dataloader_cfg to DictConfig. "
                        "It is of type: %s. Using empty DictConfig.",
                        type(converted_cfg),
                    )
                    dataloader_cfg = OmegaConf.create({})
            except Exception as e:
                log.warning(
                    "Error converting dataloader_cfg to DictConfig: %s. "
                    "Using empty DictConfig.",
                    e,
                )
                dataloader_cfg = OmegaConf.create({})

        # Ensure data_cfg is DictConfig for factory function
        if not isinstance(data_cfg, DictConfig):
            raise ValueError(
                f"Expected DictConfig for data_config, got {type(data_cfg)}"
            )

        # Create dataloaders using factory function
        dataloader_dict = create_dataloaders_from_config(
            data_config=data_cfg,
            transform_config=transform_cfg,
            dataloader_config=dataloader_cfg,
        )
        # Extract DataLoader objects from the dictionary
        train_loader = dataloader_dict["train"]["dataloader"]  # type: ignore
        val_loader = dataloader_dict["val"]["dataloader"]  # type: ignore

        log.info(
            "Data loaded successfully. Train batches: %s, Val batches: %s",
            len(train_loader),  # type: ignore
            len(val_loader),  # type: ignore
        )

        return train_loader, val_loader

    except Exception as e:
        log.error("Error loading data: %s", str(e))
        raise DataError(f"Error loading data: {str(e)}") from e
