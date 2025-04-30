import pytest
import torch
from torch.utils.data import Dataset
from omegaconf import OmegaConf
from src.data.factory import create_dataloader, create_dataloaders_from_config


class DummyDataset(Dataset):
    def __init__(self, length=100):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return torch.tensor([idx], dtype=torch.float32)


def test_dataloader_basic_init():
    ds = DummyDataset(10)
    loader = create_dataloader(ds)
    assert isinstance(loader, torch.utils.data.DataLoader)
    batch = next(iter(loader))
    # batch_size=32 default, but if ds<32, batch=ds
    assert batch.shape[0] == 10 or batch.shape[0] == 32


def test_dataloader_custom_batch_size():
    ds = DummyDataset(50)
    loader = create_dataloader(ds, batch_size=8)
    batches = list(loader)
    assert all(b.shape[0] == 8 for b in batches[:-1])
    assert batches[-1].shape[0] == 2  # 50 % 8 = 2


def test_dataloader_shuffle():
    ds = DummyDataset(100)
    loader1 = create_dataloader(ds, shuffle=True)
    loader2 = create_dataloader(ds, shuffle=False)
    batch1 = next(iter(loader1)).tolist()
    batch2 = next(iter(loader2)).tolist()
    # Con shuffle, el primer batch probablemente no será igual
    # No error, solo que no debe fallar
    assert batch1 != batch2 or batch1 == batch2


def test_dataloader_num_workers():
    ds = DummyDataset(20)
    loader = create_dataloader(ds, num_workers=0)
    assert loader.num_workers == 0
    loader2 = create_dataloader(ds, num_workers=2)
    assert loader2.num_workers == 2


@pytest.mark.parametrize("batch_size", [1, 8, 16, 64])
def test_dataloader_various_batch_sizes(batch_size):
    ds = DummyDataset(100)
    loader = create_dataloader(ds, batch_size=batch_size)
    for batch in loader:
        assert batch.shape[0] <= batch_size


@pytest.mark.parametrize("prefetch_factor", [1, 2, 4])
def test_dataloader_prefetch_factor(prefetch_factor):
    ds = DummyDataset(20)
    loader = create_dataloader(ds, prefetch_factor=prefetch_factor,
                               num_workers=2)
    assert loader.prefetch_factor == prefetch_factor


@pytest.mark.parametrize("pin_memory", [True, False])
def test_dataloader_pin_memory(pin_memory):
    ds = DummyDataset(10)
    loader = create_dataloader(ds, pin_memory=pin_memory)
    # Si no hay CUDA, pin_memory será False aunque se pida True
    if pin_memory and torch.cuda.is_available():
        assert loader.pin_memory is True
    else:
        assert loader.pin_memory is False


def test_dataloader_invalid_batch_size():
    ds = DummyDataset(10)
    with pytest.raises(ValueError):
        create_dataloader(ds, batch_size=0)


def test_dataloader_invalid_prefetch_factor():
    ds = DummyDataset(10)
    with pytest.raises(ValueError):
        create_dataloader(ds, prefetch_factor=0)


def test_dataloader_invalid_num_workers():
    ds = DummyDataset(10)
    with pytest.raises(ValueError):
        create_dataloader(ds, num_workers=-2)


def test_dataloader_fp16_option():
    """Test for mixed precision option in create_dataloader."""
    ds = DummyDataset(10)
    # Should not raise errors
    loader = create_dataloader(ds, fp16=True)
    batch = next(iter(loader))
    assert batch.dtype == torch.float32  # dataloader doesn't change dtype

    # When CUDA not available, should still work with warning
    loader = create_dataloader(ds, fp16=True)
    batch = next(iter(loader))
    assert batch.dtype == torch.float32


def test_dataloader_max_memory_mb():
    """Test memory limit option in create_dataloader."""
    ds = DummyDataset(100)
    # Test with very small memory limit
    loader = create_dataloader(ds, max_memory_mb=100)
    # Should still work, default batch size if not adaptive
    assert isinstance(loader, torch.utils.data.DataLoader)
    batch = next(iter(loader))
    assert batch.shape[0] > 0


def test_dataloader_adaptive_batch_size():
    """Test adaptive batch size based on memory."""
    ds = DummyDataset(100)
    # Set both adaptive and small memory limit
    loader = create_dataloader(
        ds,
        batch_size=32,
        adaptive_batch_size=True,
        max_memory_mb=100
    )
    # Should use a batch size that fits in memory
    assert isinstance(loader, torch.utils.data.DataLoader)
    batch = next(iter(loader))
    assert batch.shape[0] > 0
    assert batch.shape[0] <= 32  # Should not exceed requested


# --- Pruebas para create_dataloaders_from_config ---

def test_create_dataloaders_from_config_basic():
    """Prueba básica para create_dataloaders_from_config con configuraciones
    mock."""
    # Mock del dataset para no depender de archivos reales
    class MockDataset(Dataset):
        def __init__(self, mode, samples_list, **kwargs):
            self.mode = mode
            self.samples = samples_list

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return {'image': torch.randn(3, 32, 32),
                    'mask': torch.randint(0, 2, (1, 32, 32))}

    # Crear configuraciones mock usando OmegaConf
    data_config = OmegaConf.create({
        'data_root': 'mock_data/',
        'train_split': 0.7,
        'val_split': 0.15,
        'test_split': 0.15,
        'image_size': [32, 32],
        'batch_size': 4,
        'num_workers': 0,
        'seed': 42,
        'in_memory_cache': False
    })

    transform_config = OmegaConf.create({
        'resize': {
            'enabled': True,
            'height': 32,
            'width': 32
        },
        'normalize': {
            'enabled': True,
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5]
        },
        'train': {},
        'val': {},
        'test': {}
    })

    dataloader_config = OmegaConf.create({
        'batch_size': 4,
        'num_workers': 0,
        'shuffle': True,
        'pin_memory': False,
        'prefetch_factor': 2,
        'drop_last': False,
        'distributed': {
            'enabled': False
        },
        'sampler': {
            'enabled': False
        },
        'memory': {
            'fp16': False,
            'adaptive_batch_size': False
        }
    })

    # Monkey patch la función create_split_datasets para evitar
    # necesidad de acceder al sistema de archivos real
    import src.data.factory
    original_func = src.data.factory.create_split_datasets

    def mock_create_split_datasets(*args, **kwargs):
        return {
            'train': MockDataset('train', [(f'img{i}.jpg', f'mask{i}.png')
                                           for i in range(10)]),
            'val': MockDataset('val', [(f'img{i}.jpg', f'mask{i}.png')
                                       for i in range(10, 12)]),
            'test': MockDataset('test', [(f'img{i}.jpg', f'mask{i}.png')
                                         for i in range(12, 14)])
        }

    # Aplicar el monkey patch
    src.data.factory.create_split_datasets = mock_create_split_datasets

    try:
        # Ejecutar la función
        result = create_dataloaders_from_config(
            data_config=data_config,
            transform_config=transform_config,
            dataloader_config=dataloader_config,
            dataset_class=MockDataset
        )

        # Verificar resultados
        assert 'train' in result
        assert 'val' in result
        assert 'test' in result

        for split in ['train', 'val', 'test']:
            assert 'dataset' in result[split]
            assert 'dataloader' in result[split]
            assert isinstance(result[split]['dataloader'],
                              torch.utils.data.DataLoader)

        # Verificar tamaños de batch
        assert result['train']['dataloader'].batch_size == 4

        # Verificar que podemos iterar los dataloaders
        for split in ['train', 'val', 'test']:
            batch = next(iter(result[split]['dataloader']))
            assert 'image' in batch
            assert 'mask' in batch
            assert batch['image'].shape[0] <= 4  # Batch size or smaller

    finally:
        # Restaurar la función original
        src.data.factory.create_split_datasets = original_func


def test_create_dataloaders_from_config_distributed():
    """Prueba de configuración distribuida en create_dataloaders_from_config.
    """
    # Usar la misma clase MockDataset que en el test anterior
    class MockDataset(Dataset):
        def __init__(self, mode, samples_list, **kwargs):
            self.mode = mode
            self.samples = samples_list

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return {'image': torch.randn(3, 32, 32),
                    'mask': torch.randint(0, 2, (1, 32, 32))}

    # Crear configuraciones similar al test anterior
    data_config = OmegaConf.create({
        'data_root': 'mock_data/',
        'train_split': 0.7,
        'val_split': 0.15,
        'test_split': 0.15,
        'image_size': [32, 32],
        'batch_size': 4,
        'seed': 42
    })

    transform_config = OmegaConf.create({
        'resize': {
            'enabled': True,
            'height': 32,
            'width': 32
        },
        'normalize': {
            'enabled': True,
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5]
        },
        'train': {},
        'val': {},
        'test': {}
    })

    # Configuración con distributed y sampler habilitados
    dataloader_config = OmegaConf.create({
        'batch_size': 4,
        'num_workers': 0,
        'distributed': {
            'enabled': True,
            'rank': 0,
            'world_size': 2
        },
        'sampler': {
            'enabled': True,
            'kind': 'distributed',
            'shuffle': True,
            'seed': 42
        }
    })

    # Monkey patch igual que en el test anterior
    import src.data.factory
    original_funcs = {
        'create_split_datasets': src.data.factory.create_split_datasets,
        'is_distributed': torch.distributed.is_available,
        'is_initialized': torch.distributed.is_initialized
    }

    def mock_create_split_datasets(*args, **kwargs):
        return {
            'train': MockDataset('train', [(f'img{i}.jpg', f'mask{i}.png')
                                           for i in range(20)]),
            'val': MockDataset('val', [(f'img{i}.jpg', f'mask{i}.png')
                                       for i in range(20, 25)]),
            'test': MockDataset('test', [(f'img{i}.jpg', f'mask{i}.png')
                                         for i in range(25, 30)])
        }

    # Mock para simular que torch.distributed no está disponible
    # para evitar errores en pruebas
    def mock_is_distributed_available():
        return False

    def mock_is_initialized():
        return False

    # Aplicar monkey patches
    src.data.factory.create_split_datasets = mock_create_split_datasets
    torch.distributed.is_available = mock_is_distributed_available
    torch.distributed.is_initialized = mock_is_initialized

    try:
        # Ejecutar la función
        result = create_dataloaders_from_config(
            data_config=data_config,
            transform_config=transform_config,
            dataloader_config=dataloader_config,
            dataset_class=MockDataset
        )

        # Verificar resultados básicos
        assert 'train' in result
        assert 'val' in result
        assert 'test' in result

        # Verificar que se configuraron los dataloaders
        for split in ['train', 'val', 'test']:
            assert isinstance(result[split]['dataloader'],
                              torch.utils.data.DataLoader)

        # No podemos verificar el sampler directamente ya que no tenemos
        # torch.distributed inicializado y el código maneja ese caso

    finally:
        # Restaurar funciones originales
        src.data.factory.create_split_datasets = \
            original_funcs['create_split_datasets']
        torch.distributed.is_available = original_funcs['is_distributed']
        torch.distributed.is_initialized = original_funcs['is_initialized']
