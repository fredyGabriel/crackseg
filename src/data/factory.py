import os
import warnings
import torch
from torch.utils.data import DataLoader, Dataset

from .dataset import CrackSegmentationDataset
from omegaconf import DictConfig, OmegaConf
from .sampler import sampler_factory
from .memory import get_available_gpu_memory
from .splitting import create_split_datasets
from .distributed import get_rank, get_world_size


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
    sampler_config: dict = None,
    rank: int = None,
    world_size: int = None,
    fp16: bool = False,           # <-- Soporte para mixed precision
    max_memory_mb: float = None,  # <-- Control de memoria máxima
    adaptive_batch_size: bool = False,  # <-- Ajustar batch size según memoria
    **kwargs
) -> DataLoader:
    """
    Creates and configures a PyTorch DataLoader with sensible defaults.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int): How many samples per batch to load. Default: 32.
        num_workers (int): How many subprocesses to use for data loading.
        shuffle (bool): Set to True to have the data reshuffled at every epoch.
        pin_memory (bool): If True, the data loader will copy Tensors into CUDA
            pinned memory before returning them. Recommended for GPU training.
        prefetch_factor (int): Number of batches loaded in advance by each
            worker.
        sampler_config (dict, optional): Sampler configuration dict. If given,
            uses sampler_factory to create a custom sampler. Example:
            {'kind': 'distributed', 'shuffle': True, 'seed': 42, ...}.
        rank (int, optional): Distributed process rank (for distributed
            training).
        world_size (int, optional): Number of processes (for distributed
            training).
        fp16 (bool): Whether to use mixed precision (FP16) if available.
            Default: False.
        max_memory_mb (float, optional): Maximum GPU memory to use in MB.
            If None, uses all available memory.
        adaptive_batch_size (bool): Whether to adjust batch size based on
            available memory. Default: False.
        **kwargs: Additional keyword arguments to pass to the DataLoader
            constructor.

    Returns:
        DataLoader: A configured PyTorch DataLoader instance.

    Raises:
        ValueError: If batch_size or prefetch_factor are not positive,
            or if num_workers is less than -1.
        ValueError: If both shuffle and sampler are set (PyTorch limitation).

    Note:
        If using DistributedSampler, debes llamar a
        `set_epoch(epoch)` en el sampler al inicio de cada época para
        asegurar el shuffling correcto entre procesos.

        When using fp16=True, you should wrap your training loop with
        torch.cuda.amp.autocast() context manager.
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

    # --- Memory Optimization (if requested) ---
    if adaptive_batch_size and torch.cuda.is_available():
        # Calculate available memory (retain 10% for safety)
        if max_memory_mb is None:
            available_mb = get_available_gpu_memory() * 0.9
        else:
            available_mb = min(max_memory_mb, get_available_gpu_memory() * 0.9)

        # Very rough heuristic: assume 4 bytes per float * 2 for gradients and
        # optimizer
        # Multiply by batch size to get total memory
        # This is very approximate and should be replaced with proper
        # estimation
        # Placeholder - adjust based on dataset
        approx_sample_size_mb = 4 * 0.001
        max_batch_size = int(available_mb // approx_sample_size_mb)

        # Don't exceed user-specified batch size
        batch_size = min(batch_size, max_batch_size)
        batch_size = max(1, batch_size)  # Ensure at least 1

        warnings.warn(
            f"Adaptive batch size used: {batch_size} (limited by memory)"
        )

    # --- Mixed Precision ---
    if fp16 and not torch.cuda.is_available():
        warnings.warn(
            "Mixed precision (fp16) requested but CUDA not available. "
            "Falling back to standard precision."
        )
        fp16 = False

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
    can_pin_memory = pin_memory and torch.cuda.is_available()
    if pin_memory and not can_pin_memory:
        warnings.warn(
            "pin_memory=True requires CUDA availability. "
            "Setting pin_memory=False."
        )

    # --- Sampler logic ---
    sampler = None
    if sampler_config is not None:
        sampler_kind = sampler_config.get('kind')
        sampler_kwargs = dict(sampler_config)
        if sampler_kind == 'distributed':
            # Permitir override por argumentos directos
            if world_size is not None:
                sampler_kwargs['num_replicas'] = world_size
            if rank is not None:
                sampler_kwargs['rank'] = rank
        sampler = sampler_factory(
            kind=sampler_kind,
            data_source=dataset,
            labels=sampler_kwargs.get('labels'),
            indices=sampler_kwargs.get('indices'),
            replacement=sampler_kwargs.get('replacement', False),
            num_samples=sampler_kwargs.get('num_samples'),
            num_replicas=sampler_kwargs.get('num_replicas'),
            rank=sampler_kwargs.get('rank'),
            shuffle=sampler_kwargs.get('shuffle', True),
            seed=sampler_kwargs.get('seed', 0),
            drop_last=sampler_kwargs.get('drop_last', False)
        )
        if shuffle:
            warnings.warn(
                "Both sampler and shuffle are set. "
                "Setting shuffle=False (PyTorch does not allow both)."
            )
            shuffle = False

    # --- Create DataLoader ---
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=actual_num_workers,
        pin_memory=can_pin_memory,
        prefetch_factor=prefetch_factor if actual_num_workers > 0 else None,
        persistent_workers=True if actual_num_workers > 0 else False,
        **kwargs
    )

    return dataloader


def create_dataloaders_from_config(
    data_config: DictConfig,
    transform_config: DictConfig,
    dataloader_config: DictConfig,
    dataset_class=CrackSegmentationDataset
) -> dict:
    """
    Creates train/val/test datasets and dataloaders from Hydra configs.

    Args:
        data_config (DictConfig): General data configuration
                               (configs/data/default.yaml).
        transform_config (DictConfig): Transform configuration
                                    (configs/data/transform.yaml).
        dataloader_config (DictConfig): DataLoader configuration
                                     (configs/data/dataloader.yaml).
        dataset_class: Dataset class to use, defaults to
            CrackSegmentationDataset.

    Returns:
        dict: Dictionary with 'train', 'val', 'test' datasets and dataloaders.
              For example: {
                  'train': {'dataset': train_dataset,
                            'dataloader': train_loader},
                  'val': {'dataset': val_dataset, 'dataloader': val_loader},
                  'test': {'dataset': test_dataset, 'dataloader': test_loader}
              }
    """
    # Convertir y validar configuraciones
    data_cfg = OmegaConf.to_container(data_config, resolve=True)
    transform_cfg = OmegaConf.to_container(transform_config, resolve=True)
    dl_cfg = OmegaConf.to_container(dataloader_config, resolve=True)

    validate_data_config(data_cfg)
    validate_transform_config(transform_cfg)

    # Extraer parámetros de data_cfg
    data_root = data_cfg['data_root']
    image_size = data_cfg['image_size']
    in_memory_cache = data_cfg.get('in_memory_cache', False)
    seed = data_cfg.get('seed', 42)

    # Preparar ratios para splitting
    ratios = {
        'train': data_cfg['train_split'],
        'val': data_cfg['val_split'],
        'test': data_cfg['test_split']
    }

    # Obtener datasets para cada split
    try:
        split_datasets = create_split_datasets(
            data_root=data_root,
            image_size=image_size,
            ratios=ratios,
            seed=seed,
            cache_flag=in_memory_cache,
            dataset_cls=dataset_class
        )
    except (FileNotFoundError, RuntimeError) as e:
        # Si hay error en create_split_datasets, intentamos obtener samples
        # manualmente y crear cada dataset individualmente
        from .splitting import get_all_samples
        all_samples = get_all_samples(data_root)
        if not all_samples:
            raise RuntimeError(
                f"No se encontraron muestras válidas en {data_root}"
            ) from e

        from .splitting import split_indices
        indices_map = split_indices(
            num_samples=len(all_samples),
            ratios=ratios,
            seed=seed,
            shuffle=True
        )

        split_datasets = {}
        for split_name in ['train', 'val', 'test']:
            # Obtener las muestras para este split
            split_indices_list = indices_map[split_name]
            split_samples = [all_samples[i] for i in split_indices_list]

            # Crear dataset para este split
            split_datasets[split_name] = create_crackseg_dataset(
                data_cfg=data_cfg,
                transform_cfg=transform_cfg,
                mode=split_name,
                samples_list=split_samples,
                in_memory_cache=in_memory_cache
            )

    # Preparar el resultado
    result = {}

    # Configuración de entrenamiento distribuido
    is_distributed = dl_cfg.get('distributed', {}).get('enabled', False)
    rank = dl_cfg.get('distributed', {}).get('rank', 0)
    world_size = dl_cfg.get('distributed', {}).get('world_size', 1)

    # Si estamos en un entorno distribuido real, sobreescribir con valores
    # reales
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        is_distributed = True
        rank = get_rank()
        world_size = get_world_size()

    # Configuración de sampler
    sampler_config = None
    if dl_cfg.get('sampler', {}).get('enabled', False):
        sampler_config = dict(dl_cfg.get('sampler', {}))
        # Si está en modo distribuido, ajustar el sampler
        if is_distributed and sampler_config.get('kind') != 'distributed':
            warnings.warn(
                "Distributed training detected but sampler kind is not "
                "'distributed'. Switching to distributed sampler."
            )
            sampler_config['kind'] = 'distributed'

        # Eliminar la clave 'enabled' que no es parte de la config real
        if 'enabled' in sampler_config:
            del sampler_config['enabled']

    # Configuración de memoria
    memory_cfg = dl_cfg.get('memory', {})
    fp16 = memory_cfg.get('fp16', False)
    adaptive_batch_size = memory_cfg.get('adaptive_batch_size', False)
    max_memory_mb = memory_cfg.get('max_memory_mb', None)

    # Parámetros básicos del dataloader
    batch_size = dl_cfg.get('batch_size', data_cfg.get('batch_size', 8))
    num_workers = dl_cfg.get('num_workers', data_cfg.get('num_workers', -1))
    shuffle = dl_cfg.get('shuffle', True)
    pin_memory = dl_cfg.get('pin_memory', True)
    prefetch_factor = dl_cfg.get('prefetch_factor', 2)
    drop_last = dl_cfg.get('drop_last', False)

    # Crear dataloaders para cada split
    for split_name, dataset in split_datasets.items():
        # Para test y validación, desactivar shuffling
        current_shuffle = shuffle if split_name == 'train' else False

        # Crear dataloader
        dataloader = create_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=current_shuffle,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            sampler_config=sampler_config,
            rank=rank if is_distributed else None,
            world_size=world_size if is_distributed else None,
            fp16=fp16,
            max_memory_mb=max_memory_mb,
            adaptive_batch_size=adaptive_batch_size,
            drop_last=drop_last
        )

        # Guardar en el resultado
        result[split_name] = {
            'dataset': dataset,
            'dataloader': dataloader
        }

    return result
