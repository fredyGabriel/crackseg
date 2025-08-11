"""Data loading for training (consolidated)."""

from __future__ import annotations

import logging
import os
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from crackseg.data.factory import create_dataloaders_from_config
from crackseg.utils import DataError

log = logging.getLogger(__name__)


def load_data(cfg: DictConfig) -> tuple[DataLoader[Any], DataLoader[Any]]:
    log.info("Loading data...")
    try:
        data_cfg = None
        if "experiments" in cfg:
            for exp_name in cfg.experiments:
                exp_config = cfg.experiments[exp_name]
                if hasattr(exp_config, "data"):
                    data_cfg = exp_config.data
                    break
        if data_cfg is None and hasattr(cfg, "data"):
            data_cfg = cfg.data
        if data_cfg is None:
            raise AttributeError("No data configuration found in config")

        transform_cfg = getattr(data_cfg, "transform", None)
        if transform_cfg is None and "data/transform" in cfg:
            transform_cfg = cfg["data/transform"]
        if transform_cfg is None:
            log.warning("Transform config not found. Using empty config.")
            transform_cfg = OmegaConf.create({})

        data_cfg_dict = OmegaConf.to_container(data_cfg, resolve=True)
        if not isinstance(data_cfg_dict, dict):
            raise ValueError("Expected dict from data config")

        orig_cwd = hydra.utils.get_original_cwd()
        data_root = os.path.join(
            orig_cwd, data_cfg_dict.get("data_root", "data/")
        )
        data_cfg_dict["data_root"] = data_root
        data_cfg = OmegaConf.create(data_cfg_dict)

        dataloader_cfg = getattr(data_cfg, "dataloader", None)
        if dataloader_cfg is None and "data/dataloader" in cfg:
            dataloader_cfg = cfg["data/dataloader"]
        if not isinstance(dataloader_cfg, DictConfig):
            try:
                converted = OmegaConf.create(dataloader_cfg)
                dataloader_cfg = (
                    converted
                    if isinstance(converted, DictConfig)
                    else OmegaConf.create({})
                )
            except Exception:
                dataloader_cfg = OmegaConf.create({})

        dataloader_dict = create_dataloaders_from_config(
            data_config=data_cfg,
            transform_config=transform_cfg,
            dataloader_config=dataloader_cfg,
        )
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
