from typing import Any

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from crackseg.data import factory as data_factory
from crackseg.data.dataloader import DataLoaderConfig, create_dataloader
from crackseg.data.factory import create_dataloaders_from_config


class DummyDataset(Dataset[torch.Tensor]):
    def __init__(self, length: int = 100) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor([idx], dtype=torch.float32)


def test_create_dataloader_uses_default_config() -> None:
    """Verify DataLoader is created with default settings."""
    dataset = DummyDataset(10)
    loader = create_dataloader(dataset)
    assert isinstance(loader, DataLoader)
    assert loader.batch_size == 32  # Default batch_size


def test_create_dataloader_with_custom_batch_size() -> None:
    """Test creating a DataLoader with a custom batch size."""
    dataset = DummyDataset(50)
    loader = create_dataloader(dataset, batch_size=8)
    batches = list(loader)
    assert all(b.shape[0] == 8 for b in batches[:-1])
    assert batches[-1].shape[0] == 2  # 50 % 8 = 2


def test_create_dataloader_with_shuffle_enabled() -> None:
    """Test that shuffling produces different batch orders."""
    dataset = DummyDataset(100)
    config = DataLoaderConfig(shuffle=True)
    loader1 = create_dataloader(dataset, config=config)
    loader2 = create_dataloader(dataset, config=config)
    batch1 = next(iter(loader1)).tolist()
    batch2 = next(iter(loader2)).tolist()
    assert (
        batch1 != batch2
    ), "Shuffled loaders should produce different batches"


@pytest.mark.parametrize("num_workers", [0, 1, 2])
def test_create_dataloader_with_num_workers(num_workers: int) -> None:
    """Verify DataLoader is created correctly with different workers."""
    dataset = DummyDataset(20)
    config = DataLoaderConfig(num_workers=num_workers)
    loader = create_dataloader(dataset, config=config)
    assert loader.num_workers == num_workers


@pytest.mark.parametrize("batch_size", [1, 8, 16, 64])
def test_dataloader_various_batch_sizes(batch_size: int) -> None:
    """Test that the dataloader respects various batch sizes."""
    dataset = DummyDataset(100)
    loader = create_dataloader(dataset, batch_size=batch_size)
    for batch in loader:
        assert batch.shape[0] <= batch_size


@pytest.mark.parametrize("prefetch_factor", [2, 4])
def test_dataloader_prefetch_factor(prefetch_factor: int) -> None:
    """Verify DataLoader is created with a prefetch factor."""
    dataset = DummyDataset(20)
    # prefetch_factor only works for num_workers > 0
    config = DataLoaderConfig(num_workers=1, prefetch_factor=prefetch_factor)
    loader = create_dataloader(dataset, config=config)
    assert loader.prefetch_factor == prefetch_factor


@pytest.mark.parametrize("pin_memory", [True, False])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_dataloader_pin_memory(pin_memory: bool) -> None:
    """Test DataLoader with pin_memory enabled and disabled."""
    dataset = DummyDataset(10)
    config = DataLoaderConfig(pin_memory=pin_memory)
    loader = create_dataloader(dataset, config=config)
    assert loader.pin_memory == pin_memory


def test_dataloader_raises_error_for_invalid_batch_size() -> None:
    """Test that a ValueError is raised for a batch size of 0."""
    dataset = DummyDataset(10)
    with pytest.raises(
        ValueError, match="batch_size must be a positive integer"
    ):
        create_dataloader(dataset, batch_size=0)


def test_dataloader_fp16_config_is_respected() -> None:
    """Test that the fp16 configuration flag is handled correctly."""
    dataset = DummyDataset(10)
    # The function itself doesn't change tensor dtype, but we can check the
    # config path
    config = DataLoaderConfig(fp16=True)
    loader = create_dataloader(dataset, config=config)
    batch = next(iter(loader))
    assert batch.dtype == torch.float32  # Dtype should remain as created
    # No direct assertion on fp16, just that it doesn't crash


def test_dataloader_adaptive_batch_size_config() -> None:
    """Test that the adaptive batch size configuration is handled."""
    dataset = DummyDataset(100)
    config = DataLoaderConfig(adaptive_batch_size=True, max_memory_mb=1024)
    loader = create_dataloader(dataset, config=config)
    assert isinstance(loader, DataLoader)
    batch = next(iter(loader))
    assert batch.shape[0] > 0
    # Actual batch size depends on available memory, so we check it's <=
    # default
    assert batch.shape[0] <= 32


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

    original_func = data_factory.create_split_datasets

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

    data_factory.create_split_datasets = mock_create_split_datasets

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
            assert isinstance(result[split]["dataloader"], DataLoader)

        if isinstance(result["train"]["dataloader"], DataLoader):
            assert (
                result["train"]["dataloader"].batch_size == 4
            )  # noqa: PLR2004

        for split in ["train", "val", "test"]:
            batch = next(iter(result[split]["dataloader"]))
            assert "image" in batch
            assert "mask" in batch
            assert batch["image"].shape[0] <= 4  # noqa: PLR2004

    finally:
        data_factory.create_split_datasets = original_func


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

    # Mock torch.distributed functions since they may not be available
    def mock_distributed_available() -> bool:
        return False

    def mock_distributed_initialized() -> bool:
        return False

    original_funcs: dict[str, Any] = {
        "create_split_datasets": data_factory.create_split_datasets,
        "is_distributed": mock_distributed_available,
        "is_initialized": mock_distributed_initialized,
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

    data_factory.create_split_datasets = mock_create_split_datasets
    torch.distributed.is_available = mock_distributed_available  # type: ignore[attr-defined]
    torch.distributed.is_initialized = mock_distributed_initialized  # type: ignore[attr-defined]

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
            assert isinstance(result[split]["dataloader"], DataLoader)

    finally:
        data_factory.create_split_datasets = original_funcs[
            "create_split_datasets"
        ]
        torch.distributed.is_available = original_funcs["is_distributed"]  # type: ignore[attr-defined]
        torch.distributed.is_initialized = original_funcs["is_initialized"]  # type: ignore[attr-defined]
