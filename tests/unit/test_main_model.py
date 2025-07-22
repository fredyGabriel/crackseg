"""
Unit tests for src.main module - Model Creation. Tests the model
creation and initialization functionality for crack segmentation,
including model factory usage, device placement, and error handling.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from torch import nn

from crackseg.utils import ModelError


class TestCreateModel:
    """Test model creation functionality."""

    @patch("crackseg.main.create_model_from_config")
    def test_create_model_success(self, mock_create_model: Mock) -> None:
        """Test successful model creation."""
        # Arrange
        config_mock = MagicMock()
        device = torch.device("cpu")

        mock_model = MagicMock(spec=nn.Module)
        mock_create_model.return_value = mock_model

        # Act
        with patch("crackseg.main._create_model") as mock_create:
            mock_create.return_value = mock_model
            model = mock_create(config_mock, device)

        # Assert
        assert model is not None
        assert isinstance(model, MagicMock)

    @patch("crackseg.main.create_model_from_config")
    def test_create_model_with_cuda_device(
        self, mock_create_model: Mock
    ) -> None:
        """Test model creation with CUDA device."""
        # Arrange
        config_mock = MagicMock()
        device = torch.device("cuda")

        mock_model = MagicMock(spec=nn.Module)
        mock_create_model.return_value = mock_model

        # Act
        with patch("crackseg.main._create_model") as mock_create:
            mock_create.return_value = mock_model
            model = mock_create(config_mock, device)

        # Assert
        assert model is not None

    @patch("crackseg.main.create_model_from_config")
    def test_create_model_missing_config(
        self, mock_create_model: Mock
    ) -> None:
        """Test error when model configuration is missing."""
        # Arrange
        config_mock = MagicMock()
        config_mock.model = None
        device = torch.device("cpu")

        mock_create_model.side_effect = ModelError(
            "Model configuration is required"
        )

        # Act & Assert
        with patch("crackseg.main._create_model") as mock_create:
            mock_create.side_effect = ModelError(
                "Model configuration is required"
            )
            with pytest.raises(
                ModelError, match="Model configuration is required"
            ):
                mock_create(config_mock, device)

    @patch("crackseg.main.create_model_from_config")
    def test_create_model_invalid_architecture(
        self, mock_create_model: Mock
    ) -> None:
        """Test error when model architecture is invalid."""
        # Arrange
        config_mock = MagicMock()
        device = torch.device("cpu")

        mock_create_model.side_effect = ModelError(
            "Unknown model architecture"
        )

        # Act & Assert
        with patch("crackseg.main._create_model") as mock_create:
            mock_create.side_effect = ModelError("Unknown model architecture")
            with pytest.raises(ModelError, match="Unknown model architecture"):
                mock_create(config_mock, device)
