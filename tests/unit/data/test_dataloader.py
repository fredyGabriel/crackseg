import pytest
import torch
from torch.utils.data import Dataset

from crackseg.data.dataloader import create_dataloader


class DummyDataset(Dataset[torch.Tensor]):
    def __init__(self, length: int = 10):
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor([idx], dtype=torch.float32)


def test_create_dataloader_basic() -> None:
    ds = DummyDataset(5)
    loader = create_dataloader(ds)
    assert isinstance(loader, torch.utils.data.DataLoader)
    batch = next(iter(loader))
    assert batch.shape[0] == 5 or batch.shape[0] == 32  # noqa: PLR2004


def test_create_dataloader_invalid_batch_size() -> None:
    ds = DummyDataset(5)
    with pytest.raises(ValueError):
        create_dataloader(ds, batch_size=0)  # type: ignore
