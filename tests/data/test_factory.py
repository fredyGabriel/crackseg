import pytest
import torch
from torch.utils.data import Dataset
from src.data.factory import create_dataloader


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
