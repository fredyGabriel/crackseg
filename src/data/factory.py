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
