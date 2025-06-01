from collections.abc import Sized
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.data.factory import create_dataloaders_from_config
from src.utils.exceptions import DataError
from src.utils.logging import get_logger

log = get_logger("evaluation.data")


def get_evaluation_dataloader(
    config: dict[str, Any] | DictConfig,
    data_dir: str | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> DataLoader[Any]:
    """
    Get dataloader for evaluation.

    Args:
        config: Configuration dictionary or DictConfig
        data_dir: Path to data directory (overrides config)
        batch_size: Batch size (overrides config)
        num_workers: Number of workers (overrides config)

    Returns:
        DataLoader for evaluation
    """
    # Clone config to avoid modifying the original
    if isinstance(config, DictConfig):
        data_config = OmegaConf.create(
            OmegaConf.to_container(config.data, resolve=True)
        )
    else:
        data_config = OmegaConf.create(config["data"])

    # Override data directory if provided
    if data_dir is not None:
        data_config.data_root = data_dir
        log.info("Using data directory: %s", data_dir)

    # Override batch size if provided
    if batch_size is not None:
        data_config.batch_size = batch_size
        log.info("Using batch size: %d", batch_size)

    # Override num_workers if provided
    if num_workers is not None:
        data_config.num_workers = num_workers
        log.info("Using num_workers: %d", num_workers)

    # Get data loaders
    try:
        if not isinstance(data_config, DictConfig):
            data_config = OmegaConf.create(dict(data_config))
        transform_config = data_config.get("transforms", OmegaConf.create({}))
        dataloaders_dict = create_dataloaders_from_config(
            data_config=data_config,
            transform_config=transform_config,
            dataloader_config=data_config,
        )

        # We only need the test dataloader for evaluation
        test_loader = dataloaders_dict.get("test", {}).get("dataloader")

        if test_loader is None:
            raise DataError("Test dataloader could not be created")

        if not isinstance(test_loader, DataLoader):
            raise DataError("Test dataloader is not a DataLoader instance")
        log.info(
            "Test dataset loaded with %d samples",
            len(cast(Sized, test_loader.dataset)),
        )
        return test_loader

    except Exception as e:
        raise DataError(f"Error during data loading: {str(e)}") from e
