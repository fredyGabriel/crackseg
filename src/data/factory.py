import warnings
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from src.data.dataloader import DataLoaderConfig, create_dataloader
from src.data.dataset import create_crackseg_dataset
from src.data.validation import validate_data_config

from .dataset import CrackSegmentationDataset
from .distributed import get_rank, get_world_size
from .splitting import DatasetCreationConfig, create_split_datasets


def _create_or_load_split_datasets(
    data_config: DictConfig,
    transform_config: DictConfig,
    dataloader_config: DictConfig,
    dataset_class: type[CrackSegmentationDataset],
) -> dict[str, Dataset[Any]]:
    """Helper function to create or load split datasets."""
    data_root = data_config["data_root"]
    in_memory_cache = data_config.get("in_memory_cache", False)
    seed = data_config.get("seed", 42)
    ratios = {
        "train": data_config["train_split"],
        "val": data_config["val_split"],
        "test": data_config["test_split"],
    }
    max_train_samples = dataloader_config.get("max_train_samples", None)
    max_val_samples = dataloader_config.get("max_val_samples", None)
    max_test_samples = dataloader_config.get("max_test_samples", None)

    try:
        dataset_creation_cfg = DatasetCreationConfig(
            data_root=data_root,
            transform_cfg=transform_config,
            dataset_cls=dataset_class,
            seed=seed,
            cache_flag=in_memory_cache,
            max_train_samples=max_train_samples,
            max_val_samples=max_val_samples,
            max_test_samples=max_test_samples,
        )
        temp_split_datasets = create_split_datasets(
            config=dataset_creation_cfg
        )
        return cast(dict[str, Dataset[Any]], temp_split_datasets)
    except (FileNotFoundError, RuntimeError) as e:
        from .splitting import get_all_samples, split_indices

        all_samples = get_all_samples(data_root)
        if not all_samples:
            raise RuntimeError(f"No valid samples found in {data_root}") from e
        indices_map = split_indices(
            num_samples=len(all_samples),
            ratios=ratios,
            seed=seed,
            shuffle=True,
        )
        split_datasets_fallback: dict[str, Dataset[Any]] = {}
        for split_name in ["train", "val", "test"]:
            split_indices_list = indices_map[split_name]
            split_samples_list = [all_samples[i] for i in split_indices_list]
            if split_name not in transform_config:
                raise ValueError(
                    f"Transform config missing for split: {split_name}"
                ) from e
            split_transform_cfg = transform_config[split_name]
            max_samples_for_split = None
            if split_name == "train":
                max_samples_for_split = max_train_samples
            elif split_name == "val":
                max_samples_for_split = max_val_samples
            elif split_name == "test":
                max_samples_for_split = max_test_samples
            split_datasets_fallback[split_name] = create_crackseg_dataset(
                data_cfg=data_config,
                transform_cfg=split_transform_cfg,
                mode=split_name,
                samples_list=split_samples_list,
                in_memory_cache=in_memory_cache,
                max_samples=max_samples_for_split,
            )
        return split_datasets_fallback


def _prepare_dataloader_params(
    dataloader_config: DictConfig, data_config: DictConfig
) -> tuple[DataLoaderConfig, int, int, bool, int]:
    """Helper to prepare DataLoaderConfig and related parameters."""
    dl_dist = dataloader_config.get("distributed", OmegaConf.create({}))
    is_distributed = dl_dist.get("enabled", False)
    rank_val = dl_dist.get("rank", 0)
    world_size_val = dl_dist.get("world_size", 1)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        is_distributed = True
        rank_val = get_rank()
        world_size_val = get_world_size()

    sampler_config_to_pass: dict[str, Any]
    sampler_cfg_node = dataloader_config.get("sampler", OmegaConf.create({}))
    if sampler_cfg_node.get("enabled", False):
        converted_sampler_cfg = OmegaConf.to_container(
            sampler_cfg_node, resolve=True
        )
        if isinstance(converted_sampler_cfg, dict):
            sampler_config_to_pass = cast(
                dict[str, Any], converted_sampler_cfg
            )
            if (
                is_distributed
                and sampler_config_to_pass.get("kind") != "distributed"
            ):
                warnings.warn(
                    "Distributed training detected but sampler kind is not "
                    "'distributed'. Switching to distributed sampler.",
                    stacklevel=2,
                )
                sampler_config_to_pass["kind"] = "distributed"
            if "enabled" in sampler_config_to_pass:
                del sampler_config_to_pass["enabled"]
        else:
            warnings.warn(
                "sampler_cfg_node.get('sampler') did not convert to a dict. "
                "Using {}.",
                stacklevel=2,
            )
            sampler_config_to_pass = {}
    else:
        sampler_config_to_pass = {}

    memory_cfg = dataloader_config.get("memory", OmegaConf.create({}))

    # Extract parameters for DataLoaderConfig
    num_workers = dataloader_config.get(
        "num_workers", data_config.get("num_workers", -1)
    )
    shuffle = dataloader_config.get("shuffle", True)
    pin_memory = dataloader_config.get("pin_memory", True)
    prefetch_factor = dataloader_config.get("prefetch_factor", 2)
    fp16 = memory_cfg.get("fp16", False)
    max_memory_mb = memory_cfg.get("max_memory_mb", None)
    adaptive_batch_size = memory_cfg.get("adaptive_batch_size", False)
    drop_last = dataloader_config.get("drop_last", False)
    # Dataloader extra kwargs from the main dataloader_config level, not
    # memory_cfg
    dataloader_extra_kwargs = {
        k: v
        for k, v in OmegaConf.to_container(
            dataloader_config, resolve=True
        ).items()
        if k
        not in [
            "distributed",
            "sampler",
            "memory",
            "batch_size",
            "num_workers",
            "shuffle",
            "pin_memory",
            "prefetch_factor",
            "drop_last",
            "max_train_samples",
            "max_val_samples",
            "max_test_samples",
        ]
    }
    if drop_last:
        dataloader_extra_kwargs["drop_last"] = drop_last

    loader_config = DataLoaderConfig(
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        sampler_config=sampler_config_to_pass,
        rank=rank_val if is_distributed else None,
        world_size=world_size_val if is_distributed else None,
        fp16=fp16,
        max_memory_mb=max_memory_mb,
        adaptive_batch_size=adaptive_batch_size,
        dataloader_extra_kwargs=dataloader_extra_kwargs,
    )

    batch_size = dataloader_config.get(
        "batch_size", data_config.get("batch_size", 8)
    )
    return loader_config, rank_val, world_size_val, is_distributed, batch_size


def create_dataloaders_from_config(
    data_config: DictConfig,
    transform_config: DictConfig,
    dataloader_config: DictConfig,
    dataset_class: type[CrackSegmentationDataset] = CrackSegmentationDataset,
) -> dict[str, dict[str, Dataset[Any] | DataLoader[Any]]]:
    """
    Creates train/val/test datasets and dataloaders from Hydra configs.
    """
    validate_data_config(data_config)

    split_datasets = _create_or_load_split_datasets(
        data_config, transform_config, dataloader_config, dataset_class
    )

    (
        loader_config,
        rank_val,
        world_size_val,
        is_distributed,
        batch_size,
    ) = _prepare_dataloader_params(dataloader_config, data_config)

    result: dict[str, dict[str, Dataset[Any] | DataLoader[Any]]] = {}
    for split_name, dataset_instance in split_datasets.items():
        current_loader_config = OmegaConf.structured(loader_config)
        if split_name != "train":
            current_loader_config.shuffle = False

        if not is_distributed:
            current_loader_config.rank = None
            current_loader_config.world_size = None
        else:
            current_loader_config.rank = rank_val
            current_loader_config.world_size = world_size_val

        dataloader = create_dataloader(
            dataset=dataset_instance,
            batch_size=batch_size,
            config=current_loader_config,
        )
        result[split_name] = {
            "dataset": dataset_instance,
            "dataloader": dataloader,
        }
    return result
