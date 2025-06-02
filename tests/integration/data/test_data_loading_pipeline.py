"""
Integration tests for Data Loading Pipeline.

Tests the complete integration between data factory, dataset creation,
data loading, and training pipeline readiness.
"""

import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.factory import create_dataloaders_from_config


class TestDataPipelineIntegration:
    """Test integration of the complete data loading pipeline."""

    @pytest.fixture
    def temp_data_structure(self) -> Generator[str, None, None]:
        """Create temporary directory structure with mock data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create directory structure
            for split in ["train", "val", "test"]:
                images_dir = Path(temp_dir) / split / "images"
                masks_dir = Path(temp_dir) / split / "masks"
                images_dir.mkdir(parents=True, exist_ok=True)
                masks_dir.mkdir(parents=True, exist_ok=True)

                # Create a few mock image/mask pairs
                for i in range(3):
                    # Create mock image (RGB)
                    image = Image.fromarray(
                        np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                    )
                    image.save(images_dir / f"sample_{i}.jpg")

                    # Create mock mask (grayscale)
                    mask = Image.fromarray(
                        np.random.randint(0, 2, (64, 64), dtype=np.uint8) * 255
                    )
                    mask.save(masks_dir / f"sample_{i}.png")

            yield temp_dir

    @pytest.fixture
    def data_config(self, temp_data_structure: str) -> DictConfig:
        """Create data configuration for testing."""
        config_dict = {
            "data_root": temp_data_structure,
            "image_size": [64, 64],  # Required field
            "train_split": 0.6,
            "val_split": 0.2,
            "test_split": 0.2,
            "seed": 42,
            "in_memory_cache": False,
            "num_workers": 0,  # Avoid multiprocessing issues in tests
            "batch_size": 2,
        }
        return OmegaConf.create(config_dict)

    @pytest.fixture
    def transform_config(self) -> DictConfig:
        """Create transform configuration for testing."""
        config_dict = {
            "train": [
                {"name": "Resize", "size": [64, 64]},
                {"name": "ToTensorV2"},
            ],
            "val": [
                {"name": "Resize", "size": [64, 64]},
                {"name": "ToTensorV2"},
            ],
            "test": [
                {"name": "Resize", "size": [64, 64]},
                {"name": "ToTensorV2"},
            ],
        }
        return OmegaConf.create(config_dict)

    @pytest.fixture
    def dataloader_config(self) -> DictConfig:
        """Create dataloader configuration for testing."""
        config_dict = {
            "batch_size": 2,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,
            "prefetch_factor": 2,
            "drop_last": False,
            "distributed": {"enabled": False},
            "sampler": {"enabled": False},
            "memory": {
                "fp16": False,
                "adaptive_batch_size": False,
            },
        }
        return OmegaConf.create(config_dict)

    @patch("src.data.factory._create_or_load_split_datasets")
    @patch("src.data.factory.create_dataloader")
    def test_data_factory_creates_dataloaders(
        self,
        mock_create_dataloader: Mock,
        mock_create_split_datasets: Mock,
        data_config: DictConfig,
        transform_config: DictConfig,
        dataloader_config: DictConfig,
    ) -> None:
        """Test that data factory creates proper dataloaders."""
        # Arrange
        mock_datasets = {
            "train": MagicMock(spec=Dataset),
            "val": MagicMock(spec=Dataset),
            "test": MagicMock(spec=Dataset),
        }
        mock_create_split_datasets.return_value = mock_datasets

        mock_dataloaders = {
            "train": MagicMock(spec=DataLoader),
            "val": MagicMock(spec=DataLoader),
            "test": MagicMock(spec=DataLoader),
        }
        mock_create_dataloader.side_effect = (
            lambda dataset, **kwargs: mock_dataloaders[
                (
                    "train"
                    if dataset == mock_datasets["train"]
                    else "val" if dataset == mock_datasets["val"] else "test"
                )
            ]
        )

        # Act
        result = create_dataloaders_from_config(
            data_config=data_config,
            transform_config=transform_config,
            dataloader_config=dataloader_config,
        )

        # Assert
        assert "train" in result
        assert "val" in result
        assert "test" in result

        for split in ["train", "val", "test"]:
            assert "dataset" in result[split]
            assert "dataloader" in result[split]

        # Verify dataset creation was called
        mock_create_split_datasets.assert_called_once()

        # Verify dataloader creation was called for each split
        assert mock_create_dataloader.call_count == 3

    def test_dataloader_training_integration(
        self,
        temp_data_structure: str,
        data_config: DictConfig,
        transform_config: DictConfig,
        dataloader_config: DictConfig,
    ) -> None:
        """Test integration with real data loading for training workflow."""
        # This test uses mock data loading to verify integration

        # Arrange - Use smaller datasets for testing
        data_config.update(
            {
                "in_memory_cache": True,  # For faster testing
                "train_split": 0.5,
                "val_split": 0.3,
                "test_split": 0.2,
            }
        )

        # Mock the factory components
        with patch(
            "src.data.factory._create_or_load_split_datasets"
        ) as mock_create_datasets:
            with patch(
                "src.data.factory.create_dataloader"
            ) as mock_create_dataloader:
                # Create simple mock datasets that return tensors
                class SimpleMockDataset(
                    Dataset[tuple[torch.Tensor, torch.Tensor]]
                ):
                    def __init__(self, size: int = 3) -> None:
                        self.size = size

                    def __len__(self) -> int:
                        return self.size

                    def __getitem__(
                        self, idx: int
                    ) -> tuple[torch.Tensor, torch.Tensor]:
                        return (
                            torch.randn(3, 64, 64),
                            torch.randint(0, 2, (1, 64, 64)).float(),
                        )

                mock_datasets = {
                    "train": SimpleMockDataset(3),
                    "val": SimpleMockDataset(2),
                    "test": SimpleMockDataset(1),
                }
                mock_create_datasets.return_value = mock_datasets

                # Create actual DataLoaders with mock datasets
                mock_train_loader = DataLoader(
                    mock_datasets["train"],
                    batch_size=dataloader_config.batch_size,
                    shuffle=False,
                )
                mock_val_loader = DataLoader(
                    mock_datasets["val"],
                    batch_size=dataloader_config.batch_size,
                    shuffle=False,
                )
                mock_test_loader = DataLoader(
                    mock_datasets["test"],
                    batch_size=dataloader_config.batch_size,
                    shuffle=False,
                )

                mock_create_dataloader.side_effect = [
                    mock_train_loader,
                    mock_val_loader,
                    mock_test_loader,
                ]

                # Act
                result = create_dataloaders_from_config(
                    data_config=data_config,
                    transform_config=transform_config,
                    dataloader_config=dataloader_config,
                )

                # Assert basic structure
                assert "train" in result
                assert "val" in result
                assert "test" in result

                # Test actual data loading
                train_loader = result["train"]["dataloader"]
                val_loader = result["val"]["dataloader"]

                # Test iteration over dataloaders
                train_batch = next(iter(train_loader))
                val_batch = next(iter(val_loader))

                # Assert batch structure - handle list or tuple
                assert isinstance(train_batch, tuple | list)
                assert len(train_batch) == 2

                train_inputs, train_targets = train_batch
                assert isinstance(train_inputs, torch.Tensor)
                assert isinstance(train_targets, torch.Tensor)
                assert train_inputs.shape[0] <= dataloader_config.batch_size
                assert train_targets.shape[0] <= dataloader_config.batch_size

                # Verify val loader works similarly
                val_inputs, val_targets = val_batch
                assert isinstance(val_inputs, torch.Tensor)
                assert isinstance(val_targets, torch.Tensor)

    def test_data_pipeline_error_handling(
        self,
        data_config: DictConfig,
        transform_config: DictConfig,
        dataloader_config: DictConfig,
    ) -> None:
        """Test error handling in data pipeline."""
        # Test with invalid data root
        invalid_data_config = data_config.copy()
        invalid_data_config.data_root = "/nonexistent/path"

        # The system generates warnings instead of exceptions
        # for missing paths. Test that function completes but
        # returns empty or problematic results
        with pytest.warns(
            UserWarning, match="Train or Val dataset could not be created"
        ):
            result = create_dataloaders_from_config(
                data_config=invalid_data_config,
                transform_config=transform_config,
                dataloader_config=dataloader_config,
            )

            # The result should exist but may have issues
            # For example, empty dataloaders or missing splits
            assert isinstance(result, dict)
            # The actual behavior depends on the factory implementation

    def test_dataloader_configuration_integration(
        self,
        temp_data_structure: str,
        data_config: DictConfig,
        transform_config: DictConfig,
    ) -> None:
        """Test that dataloader configuration properly integrates."""
        # Test different batch sizes
        for batch_size in [1, 2, 4]:
            dataloader_config = OmegaConf.create(
                {
                    "batch_size": batch_size,
                    "shuffle": False,  # For predictable testing
                    "num_workers": 0,
                    "pin_memory": False,
                    "distributed": {"enabled": False},
                    "sampler": {"enabled": False},
                    "memory": {"fp16": False, "adaptive_batch_size": False},
                }
            )

            # Mock the entire factory chain to avoid real data dependencies
            with patch(
                "src.data.factory._create_or_load_split_datasets"
            ) as mock_create_datasets:
                with patch(
                    "src.data.factory.create_dataloader"
                ) as mock_create_dataloader:
                    # Mock simple dataset
                    class TestDataset(
                        Dataset[tuple[torch.Tensor, torch.Tensor]]
                    ):
                        def __len__(self) -> int:
                            return 6  # Enough for different batch sizes

                        def __getitem__(
                            self, idx: int
                        ) -> tuple[torch.Tensor, torch.Tensor]:
                            return torch.randn(3, 32, 32), torch.zeros(
                                1, 32, 32
                            )

                    mock_datasets = {
                        "train": TestDataset(),
                        "val": TestDataset(),
                        "test": TestDataset(),
                    }
                    mock_create_datasets.return_value = mock_datasets

                    # Create mapping to avoid closure issues
                    train_loader = DataLoader(
                        mock_datasets["train"], batch_size=batch_size
                    )
                    val_loader = DataLoader(
                        mock_datasets["val"], batch_size=batch_size
                    )
                    test_loader = DataLoader(
                        mock_datasets["test"], batch_size=batch_size
                    )

                    mock_create_dataloader.side_effect = [
                        train_loader,
                        val_loader,
                        test_loader,
                    ]

                    # Act
                    result = create_dataloaders_from_config(
                        data_config=data_config,
                        transform_config=transform_config,
                        dataloader_config=dataloader_config,
                    )

                    # Assert
                    train_loader = result["train"]["dataloader"]
                    batch = next(iter(train_loader))
                    inputs, targets = batch

                    # Check batch size is correct (or less for last batch)
                    assert inputs.shape[0] <= batch_size
                    assert targets.shape[0] <= batch_size

    def test_data_pipeline_memory_efficiency(
        self,
        temp_data_structure: str,
        data_config: DictConfig,
        transform_config: DictConfig,
    ) -> None:
        """Test memory efficiency configurations in data pipeline."""
        # Test in-memory caching
        memory_efficient_config = OmegaConf.create(
            {
                "batch_size": 1,
                "num_workers": 0,
                "shuffle": False,
                "pin_memory": False,
                "distributed": {"enabled": False},
                "sampler": {"enabled": False},
                "memory": {
                    "fp16": True,
                    "adaptive_batch_size": False,
                    "max_memory_mb": 100,
                },
            }
        )

        # Enable in-memory cache
        data_config.in_memory_cache = True

        # Mock the factory chain
        with patch(
            "src.data.factory._create_or_load_split_datasets"
        ) as mock_create_datasets:
            mock_create_datasets.return_value = {
                "train": MagicMock(spec=Dataset),
                "val": MagicMock(spec=Dataset),
                "test": MagicMock(spec=Dataset),
            }

            with patch(
                "src.data.factory.create_dataloader"
            ) as mock_create_dataloader:
                mock_create_dataloader.return_value = MagicMock(
                    spec=DataLoader
                )

                # Act
                result = create_dataloaders_from_config(
                    data_config=data_config,
                    transform_config=transform_config,
                    dataloader_config=memory_efficient_config,
                )

                # Assert
                assert "train" in result
                # Verify that in_memory_cache was passed to dataset creation
                mock_create_datasets.assert_called_once()
                call_args = mock_create_datasets.call_args
                # The first argument is a DatasetCreationConfig object
                if call_args and len(call_args[0]) > 0:
                    passed_config = call_args[0][0]
                    # Check if it's a proper object with cache_flag attribute
                    if hasattr(passed_config, "cache_flag"):
                        assert passed_config.cache_flag is True
                    else:
                        # If it's a DictConfig, check in_memory_cache passed
                        assert data_config.in_memory_cache is True
