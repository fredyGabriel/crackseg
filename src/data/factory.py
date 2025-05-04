import warnings
import torch
from omegaconf import DictConfig, OmegaConf
from .splitting import create_split_datasets
from .distributed import get_rank, get_world_size
from src.data.validation import validate_data_config
from src.data.dataloader import create_dataloader

from .dataset import CrackSegmentationDataset
from src.data.dataset import create_crackseg_dataset


def create_dataloaders_from_config(
    data_config: DictConfig,
    transform_config: DictConfig,
    dataloader_config: DictConfig,
    dataset_class=CrackSegmentationDataset
) -> dict:
    """
    Creates train/val/test datasets and dataloaders from Hydra configs.
    """
    # Convertir y validar configuraciones
    data_cfg = OmegaConf.to_container(data_config, resolve=True)
    transform_cfg = OmegaConf.to_container(transform_config, resolve=True)
    dl_cfg = OmegaConf.to_container(dataloader_config, resolve=True)

    validate_data_config(data_cfg)

    # Extraer parámetros generales
    data_root = data_cfg['data_root']
    in_memory_cache = data_cfg.get('in_memory_cache', False)
    seed = data_cfg.get('seed', 42)
    ratios = {
        'train': data_cfg['train_split'],
        'val': data_cfg['val_split'],
        'test': data_cfg['test_split']
    }
    max_train_samples = dl_cfg.get('max_train_samples', None)
    max_val_samples = dl_cfg.get('max_val_samples', None)
    max_test_samples = dl_cfg.get('max_test_samples', None)

    # Intentar crear splits usando create_split_datasets
    try:
        split_datasets = create_split_datasets(
            data_root=data_root,
            transform_cfg=transform_cfg,
            seed=seed,
            cache_flag=in_memory_cache,
            dataset_cls=dataset_class,
            max_train_samples=max_train_samples,
            max_val_samples=max_val_samples,
            max_test_samples=max_test_samples
        )
    except (FileNotFoundError, RuntimeError) as e:
        from .splitting import get_all_samples, split_indices
        all_samples = get_all_samples(data_root)
        if not all_samples:
            raise RuntimeError(
                f"No se encontraron muestras válidas en {data_root}"
            ) from e
        indices_map = split_indices(
            num_samples=len(all_samples),
            ratios=ratios,
            seed=seed,
            shuffle=True
        )
        split_datasets = {}
        for split_name in ['train', 'val', 'test']:
            split_indices_list = indices_map[split_name]
            split_samples = [all_samples[i] for i in split_indices_list]
            if split_name not in transform_cfg:
                raise ValueError(f"Transform config missing for split: \
{split_name}")
            split_transform_cfg = transform_cfg[split_name]
            max_samples = None
            if split_name == 'train':
                max_samples = max_train_samples
            elif split_name == 'val':
                max_samples = max_val_samples
            elif split_name == 'test':
                max_samples = max_test_samples
            split_datasets[split_name] = create_crackseg_dataset(
                data_cfg=data_cfg,
                transform_cfg=split_transform_cfg,
                mode=split_name,
                samples_list=split_samples,
                in_memory_cache=in_memory_cache,
                max_samples=max_samples
            )

    # Configuración de entrenamiento distribuido
    dl_dist = dl_cfg.get('distributed', {})
    is_distributed = dl_dist.get('enabled', False)
    rank = dl_dist.get('rank', 0)
    world_size = dl_dist.get('world_size', 1)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        is_distributed = True
        rank = get_rank()
        world_size = get_world_size()

    # Configuración de sampler
    sampler_config = None
    if dl_cfg.get('sampler', {}).get('enabled', False):
        sampler_config = dict(dl_cfg.get('sampler', {}))
        if is_distributed and sampler_config.get('kind') != 'distributed':
            warnings.warn(
                "Distributed training detected but sampler kind is not "
                "'distributed'. Switching to distributed sampler."
            )
            sampler_config['kind'] = 'distributed'
        if 'enabled' in sampler_config:
            del sampler_config['enabled']

    # Configuración de memoria y parámetros básicos
    memory_cfg = dl_cfg.get('memory', {})
    fp16 = memory_cfg.get('fp16', False)
    adaptive_batch_size = memory_cfg.get('adaptive_batch_size', False)
    max_memory_mb = memory_cfg.get('max_memory_mb', None)
    batch_size = dl_cfg.get('batch_size', data_cfg.get('batch_size', 8))
    num_workers = dl_cfg.get('num_workers', data_cfg.get('num_workers', -1))
    shuffle = dl_cfg.get('shuffle', True)
    pin_memory = dl_cfg.get('pin_memory', True)
    prefetch_factor = dl_cfg.get('prefetch_factor', 2)
    drop_last = dl_cfg.get('drop_last', False)

    # Crear dataloaders para cada split usando las funciones importadas
    result = {}
    for split_name, dataset in split_datasets.items():
        current_shuffle = shuffle if split_name == 'train' else False
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
        result[split_name] = {
            'dataset': dataset,
            'dataloader': dataloader
        }
    return result
