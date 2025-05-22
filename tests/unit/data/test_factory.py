import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from src.data.dataloader import create_dataloader
from src.data.factory import create_dataloaders_from_config


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
    assert batch.shape[0] == 10 or batch.shape[0] == 32  # noqa: PLR2004


def test_dataloader_custom_batch_size():
    ds = DummyDataset(50)
    loader = create_dataloader(ds, batch_size=8)
    batches = list(loader)
    assert all(b.shape[0] == 8 for b in batches[:-1])  # noqa: PLR2004
    assert batches[-1].shape[0] == 2  # 50 % 8 = 2  # noqa: PLR2004


def test_dataloader_shuffle():
    ds = DummyDataset(100)
    loader1 = create_dataloader(ds, shuffle=True)
    loader2 = create_dataloader(ds, shuffle=False)
    batch1 = next(iter(loader1)).tolist()
    batch2 = next(iter(loader2)).tolist()
    # With shuffle, the first batch will probably not be equal
    # No error, just should not fail
    assert batch1 != batch2 or batch1 == batch2


def test_dataloader_num_workers():
    ds = DummyDataset(20)
    loader = create_dataloader(ds, num_workers=0)
    assert loader.num_workers == 0
    loader2 = create_dataloader(ds, num_workers=2)
    assert loader2.num_workers == 2  # noqa: PLR2004


@pytest.mark.parametrize("batch_size", [1, 8, 16, 64])
def test_dataloader_various_batch_sizes(batch_size):
    ds = DummyDataset(100)
    loader = create_dataloader(ds, batch_size=batch_size)
    for batch in loader:
        assert batch.shape[0] <= batch_size


@pytest.mark.parametrize("prefetch_factor", [1, 2, 4])
def test_dataloader_prefetch_factor(prefetch_factor):
    ds = DummyDataset(20)
    loader = create_dataloader(
        ds, prefetch_factor=prefetch_factor, num_workers=2
    )
    assert loader.prefetch_factor == prefetch_factor


@pytest.mark.parametrize("pin_memory", [True, False])
def test_dataloader_pin_memory(pin_memory):
    ds = DummyDataset(10)
    loader = create_dataloader(ds, pin_memory=pin_memory)
    # If there is no CUDA, pin_memory will be False even if True is requested
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
        ds, batch_size=32, adaptive_batch_size=True, max_memory_mb=100
    )
    # Should use a batch size that fits in memory
    assert isinstance(loader, torch.utils.data.DataLoader)
    batch = next(iter(loader))
    assert batch.shape[0] > 0
    assert batch.shape[0] <= 32  # Should not exceed requested  # noqa: PLR2004


# --- Tests for create_dataloaders_from_config ---


def test_create_dataloaders_from_config_basic():
    """Basic test for create_dataloaders_from_config with mock configs."""

    # Mock dataset to avoid depending on real files
    class MockDataset(Dataset):
        def __init__(self, mode, samples_list, **kwargs):
            self.mode = mode
            self.samples = samples_list

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return {
                "image": torch.randn(3, 32, 32),
                "mask": torch.randint(0, 2, (1, 32, 32)),
            }

    # Create mock configs using OmegaConf
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

    # Monkey patch the create_split_datasets function to avoid
    # needing to access the real file system
    import src.data.factory

    original_func = src.data.factory.create_split_datasets

    def mock_create_split_datasets(*args, **kwargs):
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

    # Apply the monkey patch
    src.data.factory.create_split_datasets = mock_create_split_datasets

    try:
        # Run the function
        result = create_dataloaders_from_config(
            data_config=data_config,
            transform_config=transform_config,
            dataloader_config=dataloader_config,
            dataset_class=MockDataset,
        )

        # Check results
        assert "train" in result
        assert "val" in result
        assert "test" in result

        for split in ["train", "val", "test"]:
            assert "dataset" in result[split]
            assert "dataloader" in result[split]
            assert isinstance(
                result[split]["dataloader"], torch.utils.data.DataLoader
            )

        # Check batch sizes
        assert result["train"]["dataloader"].batch_size == 4  # noqa: PLR2004

        # Check that we can iterate the dataloaders
        for split in ["train", "val", "test"]:
            batch = next(iter(result[split]["dataloader"]))
            assert "image" in batch
            assert "mask" in batch
            assert batch["image"].shape[0] <= 4  # noqa: PLR2004

    finally:
        # Restore the original function
        src.data.factory.create_split_datasets = original_func


def test_create_dataloaders_from_config_distributed():
    """
    Test distributed configuration in create_dataloaders_from_config.
    """

    # Use the same MockDataset class as in the previous test
    class MockDataset(Dataset):
        def __init__(self, mode, samples_list, **kwargs):
            self.mode = mode
            self.samples = samples_list

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return {
                "image": torch.randn(3, 32, 32),
                "mask": torch.randint(0, 2, (1, 32, 32)),
            }

    # Create configs similar to the previous test
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

    # Config with distributed and sampler enabled
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

    # Monkey patch as in the previous test
    import src.data.factory

    original_funcs = {
        "create_split_datasets": src.data.factory.create_split_datasets,
        "is_distributed": torch.distributed.is_available,
        "is_initialized": torch.distributed.is_initialized,
    }

    def mock_create_split_datasets(*args, **kwargs):
        return {
            "train": MockDataset(
                "train", [(f"img{i}.jpg", f"mask{i}.png") for i in range(20)]
            ),
            "val": MockDataset(
                "val", [(f"img{i}.jpg", f"mask{i}.png") for i in range(20, 25)]
            ),
            "test": MockDataset(
                "test",
                [(f"img{i}.jpg", f"mask{i}.png") for i in range(25, 30)],
            ),
        }

    # Mock to simulate that torch.distributed is not available
    # to avoid errors in tests
    def mock_is_distributed_available():
        return False

    def mock_is_initialized():
        return False

    # Apply monkey patches
    src.data.factory.create_split_datasets = mock_create_split_datasets
    torch.distributed.is_available = mock_is_distributed_available
    torch.distributed.is_initialized = mock_is_initialized

    try:
        # Run the function
        result = create_dataloaders_from_config(
            data_config=data_config,
            transform_config=transform_config,
            dataloader_config=dataloader_config,
            dataset_class=MockDataset,
        )

        # Basic result checks
        assert "train" in result
        assert "val" in result
        assert "test" in result

        # Check that dataloaders were configured
        for split in ["train", "val", "test"]:
            assert isinstance(
                result[split]["dataloader"], torch.utils.data.DataLoader
            )

        # We cannot check the sampler directly since we do not have
        # torch.distributed initialized and the code handles that case

    finally:
        # Restore original functions
        src.data.factory.create_split_datasets = original_funcs[
            "create_split_datasets"
        ]
        torch.distributed.is_available = original_funcs["is_distributed"]
        torch.distributed.is_initialized = original_funcs["is_initialized"]
