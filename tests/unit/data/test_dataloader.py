import pytest
import torch
from torch.utils.data import Dataset

from src.data.dataloader import create_dataloader


class DummyDataset(Dataset):
    def __init__(self, length=10):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return torch.tensor([idx], dtype=torch.float32)


def test_create_dataloader_basic():
    ds = DummyDataset(5)
    loader = create_dataloader(ds)
    assert isinstance(loader, torch.utils.data.DataLoader)
    batch = next(iter(loader))
    assert batch.shape[0] == 5 or batch.shape[0] == 32  # noqa: PLR2004


def test_create_dataloader_invalid_batch_size():
    ds = DummyDataset(5)
    with pytest.raises(ValueError):
        create_dataloader(ds, batch_size=0)  # type: ignore
