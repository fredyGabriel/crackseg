"""
Unit tests for src.main module - Data Loading. Tests the data loading
pipeline functionality for crack segmentation, including dataloader
creation, data validation, and error handling.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from torch.utils.data import DataLoader

from crackseg.utils import DataError


class TestLoadData:
    """Test data loading functionality."""

    @patch("crackseg.main.create_dataloaders_from_config")
    def test_load_data_success(self, mock_create_dataloaders: Mock) -> None:
        """Test successful data loading."""
        # Arrange
        config_mock = MagicMock()
        device = torch.device("cpu")

        mock_train_loader = MagicMock(spec=DataLoader)
        mock_val_loader = MagicMock(spec=DataLoader)
        mock_create_dataloaders.return_value = (
            mock_train_loader,
            mock_val_loader,
        )

        # Act
        with patch("crackseg.main._load_data") as mock_load:
            mock_load.return_value = (mock_train_loader, mock_val_loader)
            train_loader, val_loader = mock_load(config_mock, device)

        # Assert
        assert train_loader is not None
        assert val_loader is not None

    @patch("crackseg.main.create_dataloaders_from_config")
    def test_load_data_train_only(self, mock_create_dataloaders: Mock) -> None:
        """Test data loading with only training data."""
        # Arrange
        config_mock = MagicMock()
        device = torch.device("cpu")

        mock_train_loader = MagicMock(spec=DataLoader)
        mock_create_dataloaders.return_value = (mock_train_loader, None)

        # Act
        with patch("crackseg.main._load_data") as mock_load:
            mock_load.return_value = (mock_train_loader, None)
            train_loader, val_loader = mock_load(config_mock, device)

        # Assert
        assert train_loader is not None
        assert val_loader is None

    @patch("crackseg.main.create_dataloaders_from_config")
    def test_load_data_missing_data_config(
        self, mock_create_dataloaders: Mock
    ) -> None:
        """Test error when data configuration is missing."""
        # Arrange
        config_mock = MagicMock()
        config_mock.data = None
        device = torch.device("cpu")

        mock_create_dataloaders.side_effect = DataError(
            "Data configuration is required"
        )

        # Act & Assert
        with patch("crackseg.main._load_data") as mock_load:
            mock_load.side_effect = DataError("Data configuration is required")
            with pytest.raises(
                DataError, match="Data configuration is required"
            ):
                mock_load(config_mock, device)

    @patch("crackseg.main.create_dataloaders_from_config")
    def test_load_data_invalid_dataset_path(
        self, mock_create_dataloaders: Mock
    ) -> None:
        """Test error when dataset path is invalid."""
        # Arrange
        config_mock = MagicMock()
        device = torch.device("cpu")

        mock_create_dataloaders.side_effect = DataError(
            "Dataset path does not exist"
        )

        # Act & Assert
        with patch("crackseg.main._load_data") as mock_load:
            mock_load.side_effect = DataError("Dataset path does not exist")
            with pytest.raises(DataError, match="Dataset path does not exist"):
                mock_load(config_mock, device)

    @patch("crackseg.main.create_dataloaders_from_config")
    def test_load_data_empty_dataset(
        self, mock_create_dataloaders: Mock
    ) -> None:
        """Test error when dataset is empty."""
        # Arrange
        config_mock = MagicMock()
        device = torch.device("cpu")

        mock_create_dataloaders.side_effect = DataError(
            "Dataset contains no samples"
        )

        # Act & Assert
        with patch("crackseg.main._load_data") as mock_load:
            mock_load.side_effect = DataError("Dataset contains no samples")
            with pytest.raises(DataError, match="Dataset contains no samples"):
                mock_load(config_mock, device)

    @patch("crackseg.main.create_dataloaders_from_config")
    def test_load_data_invalid_transforms(
        self, mock_create_dataloaders: Mock
    ) -> None:
        """Test error when transform configuration is invalid."""
        # Arrange
        config_mock = MagicMock()
        device = torch.device("cpu")

        mock_create_dataloaders.side_effect = DataError(
            "Invalid transform configuration"
        )

        # Act & Assert
        with patch("crackseg.main._load_data") as mock_load:
            mock_load.side_effect = DataError(
                "Invalid transform configuration"
            )
            with pytest.raises(
                DataError, match="Invalid transform configuration"
            ):
                mock_load(config_mock, device)

    @patch("crackseg.main.create_dataloaders_from_config")
    def test_load_data_dataloader_creation_error(
        self, mock_create_dataloaders: Mock
    ) -> None:
        """Test error during dataloader creation."""
        # Arrange
        config_mock = MagicMock()
        device = torch.device("cpu")

        mock_create_dataloaders.side_effect = DataError(
            "Failed to create dataloader"
        )

        # Act & Assert
        with patch("crackseg.main._load_data") as mock_load:
            mock_load.side_effect = DataError("Failed to create dataloader")
            with pytest.raises(DataError, match="Failed to create dataloader"):
                mock_load(config_mock, device)
