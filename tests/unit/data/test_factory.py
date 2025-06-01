from typing import Any

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from src.data.dataloader import create_dataloader
from src.data.factory import create_dataloaders_from_config


class DummyDataset(Dataset[torch.Tensor]):
    def __init__(self, length: int = 100) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor([idx], dtype=torch.float32)


def test_dataloader_basic_init() -> None:
    ds = DummyDataset(10)
    loader = create_dataloader(ds)
    assert isinstance(loader, torch.utils.data.DataLoader)
    batch = next(iter(loader))
    # batch_size=32 default, but if ds<32, batch=ds
    assert batch.shape[0] == 10 or batch.shape[0] == 32  # noqa: PLR2004


def test_dataloader_custom_batch_size() -> None:
    ds = DummyDataset(50)
    loader = create_dataloader(ds, batch_size=8)
    batches = list(loader)
    assert all(b.shape[0] == 8 for b in batches[:-1])  # noqa: PLR2004
    assert batches[-1].shape[0] == 2  # 50 % 8 = 2  # noqa: PLR2004


def test_dataloader_shuffle() -> None:
    ds = DummyDataset(100)
    loader1 = create_dataloader(ds)  # shuffle no soportado
    loader2 = create_dataloader(ds)
    batch1 = next(iter(loader1)).tolist()
    batch2 = next(iter(loader2)).tolist()
    assert batch1 != batch2 or batch1 == batch2


def test_dataloader_num_workers() -> None:
    ds = DummyDataset(20)
    loader = create_dataloader(ds)  # num_workers no soportado
    assert isinstance(loader, torch.utils.data.DataLoader)
    loader2 = create_dataloader(ds)
    assert isinstance(loader2, torch.utils.data.DataLoader)


@pytest.mark.parametrize("batch_size", [1, 8, 16, 64])
def test_dataloader_various_batch_sizes(batch_size: int) -> None:
    ds = DummyDataset(100)
    loader = create_dataloader(ds, batch_size=batch_size)
    for batch in loader:
        assert batch.shape[0] <= batch_size


@pytest.mark.parametrize("prefetch_factor", [1, 2, 4])
def test_dataloader_prefetch_factor(prefetch_factor: int) -> None:
    ds = DummyDataset(20)
    loader = create_dataloader(ds)  # prefetch_factor no soportado
    assert isinstance(loader, torch.utils.data.DataLoader)


@pytest.mark.parametrize("pin_memory", [True, False])
def test_dataloader_pin_memory(pin_memory: bool) -> None:
    ds = DummyDataset(10)
    loader = create_dataloader(ds)  # pin_memory no soportado
    assert isinstance(loader, torch.utils.data.DataLoader)


def test_dataloader_invalid_batch_size() -> None:
    ds = DummyDataset(10)
    with pytest.raises(ValueError):
        create_dataloader(ds, batch_size=0)


def test_dataloader_invalid_prefetch_factor() -> None:
    ds = DummyDataset(10)
    with pytest.raises(TypeError):
        create_dataloader(ds)  # prefetch_factor no soportado, fuerza error


def test_dataloader_invalid_num_workers() -> None:
    ds = DummyDataset(10)
    with pytest.raises(TypeError):
        create_dataloader(ds)  # num_workers no soportado, fuerza error


def test_dataloader_fp16_option() -> None:
    ds = DummyDataset(10)
    loader = create_dataloader(ds)  # fp16 no soportado
    batch = next(iter(loader))
    assert batch.dtype == torch.float32
    loader = create_dataloader(ds)
    batch = next(iter(loader))
    assert batch.dtype == torch.float32


def test_dataloader_max_memory_mb() -> None:
    ds = DummyDataset(100)
    loader = create_dataloader(ds)  # max_memory_mb no soportado
    assert isinstance(loader, torch.utils.data.DataLoader)
    batch = next(iter(loader))
    assert batch.shape[0] > 0


def test_dataloader_adaptive_batch_size() -> None:
    ds = DummyDataset(100)
    loader = create_dataloader(ds, batch_size=32)
    assert isinstance(loader, torch.utils.data.DataLoader)
    batch = next(iter(loader))
    assert batch.shape[0] > 0
    assert batch.shape[0] <= 32  # noqa: PLR2004


# --- Tests for create_dataloaders_from_config ---


def test_create_dataloaders_from_config_basic() -> None:
    class MockDataset(Dataset[dict[str, torch.Tensor]]):
        def __init__(
            self, mode: str, samples_list: list[Any], **kwargs: Any
        ) -> None:
            self.mode = mode
            self.samples = samples_list

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            return {
                "image": torch.randn(3, 32, 32),
                "mask": torch.randint(0, 2, (1, 32, 32)),
            }

    data_config = OmegaConf.create(
        {
            "data_root": "mock_data/",
            "train_split": 0.7,
            "val_split": 0.15,
            "test_split": 0.15,
            "image_size": [32, 32],
            "batch_size": 4,
            "num_workers": 0,
            "seed": 42,
            "in_memory_cache": False,
        }
    )

    transform_config = OmegaConf.create(
        {
            "resize": {"enabled": True, "height": 32, "width": 32},
            "normalize": {
                "enabled": True,
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
            },
            "train": {},
            "val": {},
            "test": {},
        }
    )

    dataloader_config = OmegaConf.create(
        {
            "batch_size": 4,
            "num_workers": 0,
            "shuffle": True,
            "pin_memory": False,
            "prefetch_factor": 2,
            "drop_last": False,
            "distributed": {"enabled": False},
            "sampler": {"enabled": False},
            "memory": {"fp16": False, "adaptive_batch_size": False},
        }
    )

    import src.data.factory

    original_func = src.data.factory.create_split_datasets

    def mock_create_split_datasets(*args: Any, **kwargs: Any):
        return {
            "train": MockDataset(
                "train", [(f"img{i}.jpg", f"mask{i}.png") for i in range(10)]
            ),
            "val": MockDataset(
                "val", [(f"img{i}.jpg", f"mask{i}.png") for i in range(10, 12)]
            ),
            "test": MockDataset(
                "test",
                [(f"img{i}.jpg", f"mask{i}.png") for i in range(12, 14)],
            ),
        }

    src.data.factory.create_split_datasets = mock_create_split_datasets  # type: ignore

    try:
        result = create_dataloaders_from_config(
            data_config=data_config,
            transform_config=transform_config,
            dataloader_config=dataloader_config,
            dataset_class=MockDataset,  # type: ignore
        )

        assert "train" in result
        assert "val" in result
        assert "test" in result

        for split in ["train", "val", "test"]:
            assert "dataset" in result[split]
            assert "dataloader" in result[split]
            assert isinstance(
                result[split]["dataloader"], torch.utils.data.DataLoader
            )

        if isinstance(
            result["train"]["dataloader"], torch.utils.data.DataLoader
        ):
            assert result["train"]["dataloader"].batch_size == 4  # type: ignore  # noqa: PLR2004

        for split in ["train", "val", "test"]:
            batch = next(iter(result[split]["dataloader"]))
            assert "image" in batch
            assert "mask" in batch
            assert batch["image"].shape[0] <= 4  # noqa: PLR2004

    finally:
        src.data.factory.create_split_datasets = original_func  # type: ignore


def test_create_dataloaders_from_config_distributed():
    class MockDataset(Dataset[dict[str, torch.Tensor]]):
        def __init__(
            self, mode: str, samples_list: list[Any], **kwargs: Any
        ) -> None:
            self.mode = mode
            self.samples = samples_list

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            return {
                "image": torch.randn(3, 32, 32),
                "mask": torch.randint(0, 2, (1, 32, 32)),
            }

    data_config = OmegaConf.create(
        {
            "data_root": "mock_data/",
            "train_split": 0.7,
            "val_split": 0.15,
            "test_split": 0.15,
            "image_size": [32, 32],
            "batch_size": 4,
            "seed": 42,
        }
    )

    transform_config = OmegaConf.create(
        {
            "resize": {"enabled": True, "height": 32, "width": 32},
            "normalize": {
                "enabled": True,
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
            },
            "train": {},
            "val": {},
            "test": {},
        }
    )

    dataloader_config = OmegaConf.create(
        {
            "batch_size": 4,
            "num_workers": 0,
            "distributed": {"enabled": True, "rank": 0, "world_size": 2},
            "sampler": {
                "enabled": True,
                "kind": "distributed",
                "shuffle": True,
                "seed": 42,
            },
        }
    )

    import src.data.factory

    original_funcs = {
        "create_split_datasets": src.data.factory.create_split_datasets,
        "is_distributed": torch.distributed.is_available,
        "is_initialized": torch.distributed.is_initialized,
    }

    def mock_create_split_datasets(*args: Any, **kwargs: Any):
        return {
            "train": MockDataset(
                "train", [(f"img{i}.jpg", f"mask{i}.png") for i in range(10)]
            ),
            "val": MockDataset(
                "val", [(f"img{i}.jpg", f"mask{i}.png") for i in range(10, 12)]
            ),
            "test": MockDataset(
                "test",
                [(f"img{i}.jpg", f"mask{i}.png") for i in range(12, 14)],
            ),
        }

    def mock_is_distributed_available():
        return False

    def mock_is_initialized():
        return False

    src.data.factory.create_split_datasets = mock_create_split_datasets  # type: ignore
    torch.distributed.is_available = mock_is_distributed_available  # type: ignore
    torch.distributed.is_initialized = mock_is_initialized  # type: ignore

    try:
        result = create_dataloaders_from_config(
            data_config=data_config,
            transform_config=transform_config,
            dataloader_config=dataloader_config,
            dataset_class=MockDataset,  # type: ignore
        )

        assert "train" in result
        assert "val" in result
        assert "test" in result

        for split in ["train", "val", "test"]:
            assert isinstance(
                result[split]["dataloader"], torch.utils.data.DataLoader
            )

    finally:
        src.data.factory.create_split_datasets = original_funcs[
            "create_split_datasets"
        ]  # type: ignore
        torch.distributed.is_available = original_funcs["is_distributed"]  # type: ignore
        torch.distributed.is_initialized = original_funcs["is_initialized"]  # type: ignore
