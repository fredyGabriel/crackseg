from typing import Dict, Any, Optional, Union
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader
from src.utils.exceptions import DataError
from src.data.factory import create_dataloaders_from_config
from src.utils.logging import get_logger

log = get_logger("evaluation.data")


def get_evaluation_dataloader(
    config: Union[Dict[str, Any], DictConfig],
    data_dir: Optional[str] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None
) -> DataLoader:
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
        data_config = OmegaConf.create(config['data'])

    # Override data directory if provided
    if data_dir is not None:
        data_config.data_root = data_dir
        log.info(f"Using data directory: {data_dir}")

    # Override batch size if provided
    if batch_size is not None:
        data_config.batch_size = batch_size
        log.info(f"Using batch size: {batch_size}")

    # Override num_workers if provided
    if num_workers is not None:
        data_config.num_workers = num_workers
        log.info(f"Using num_workers: {num_workers}")

    # Get data loaders
    try:
        transform_config = data_config.get("transforms", OmegaConf.create({}))
        dataloaders_dict = create_dataloaders_from_config(
            data_config=data_config,
            transform_config=transform_config,
            dataloader_config=data_config
        )

        # We only need the test dataloader for evaluation
        test_loader = dataloaders_dict.get('test', {}).get('dataloader')

        if test_loader is None:
            raise DataError("Test dataloader could not be created")

        log.info(
            f"Test dataset loaded with {len(test_loader.dataset)} samples"
        )
        return test_loader

    except Exception as e:
        raise DataError(f"Error during data loading: {str(e)}") from e
