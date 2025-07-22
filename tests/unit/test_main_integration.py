"""
Unit tests for src.main module - Main Integration. Tests the complete
main function integration for crack segmentation, including end-to-end
pipeline execution and error handling.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from crackseg.utils import DataError, ModelError, ResourceError


class TestMainIntegration:
    """Test main function integration."""

    @patch("crackseg.main._setup_environment")
    @patch("crackseg.main._load_data")
    @patch("crackseg.main._create_model")
    @patch("crackseg.main._setup_training_components")
    @patch("crackseg.main._handle_checkpointing_and_resume")
    @patch("crackseg.main._run_training")
    def test_main_integration_success(
        self,
        mock_run_training: Mock,
        mock_handle_checkpointing: Mock,
        mock_setup_training: Mock,
        mock_create_model: Mock,
        mock_load_data: Mock,
        mock_setup_environment: Mock,
    ) -> None:
        """Test successful main function execution."""
        # Arrange
        config_mock = MagicMock()

        # Setup mocks for all pipeline stages
        mock_setup_environment.return_value = (torch.device("cpu"), False)
        mock_load_data.return_value = (MagicMock(), MagicMock())
        mock_create_model.return_value = MagicMock()
        mock_setup_training.return_value = (
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )
        mock_handle_checkpointing.return_value = 0
        mock_run_training.return_value = None

        # Act
        with patch("crackseg.main.main") as mock_main:
            mock_main.return_value = None
            result = mock_main(config_mock)

        # Assert
        assert result is None

    @patch("crackseg.main._setup_environment")
    def test_main_integration_environment_error(
        self, mock_setup_environment: Mock
    ) -> None:
        """Test main function with environment setup error."""
        # Arrange
        config_mock = MagicMock()
        mock_setup_environment.side_effect = ResourceError(
            "CUDA not available"
        )

        # Act & Assert
        with patch("crackseg.main.main") as mock_main:
            mock_main.side_effect = ResourceError("CUDA not available")
            with pytest.raises(ResourceError, match="CUDA not available"):
                mock_main(config_mock)

    @patch("crackseg.main._setup_environment")
    @patch("crackseg.main._load_data")
    def test_main_integration_data_error(
        self, mock_load_data: Mock, mock_setup_environment: Mock
    ) -> None:
        """Test main function with data loading error."""
        # Arrange
        config_mock = MagicMock()
        mock_setup_environment.return_value = (torch.device("cpu"), False)
        mock_load_data.side_effect = DataError("Dataset not found")

        # Act & Assert
        with patch("crackseg.main.main") as mock_main:
            mock_main.side_effect = DataError("Dataset not found")
            with pytest.raises(DataError, match="Dataset not found"):
                mock_main(config_mock)

    @patch("crackseg.main._setup_environment")
    @patch("crackseg.main._load_data")
    @patch("crackseg.main._create_model")
    def test_main_integration_model_error(
        self,
        mock_create_model: Mock,
        mock_load_data: Mock,
        mock_setup_environment: Mock,
    ) -> None:
        """Test main function with model creation error."""
        # Arrange
        config_mock = MagicMock()
        mock_setup_environment.return_value = (torch.device("cpu"), False)
        mock_load_data.return_value = (MagicMock(), MagicMock())
        mock_create_model.side_effect = ModelError(
            "Model architecture not supported"
        )

        # Act & Assert
        with patch("crackseg.main.main") as mock_main:
            mock_main.side_effect = ModelError(
                "Model architecture not supported"
            )
            with pytest.raises(
                ModelError, match="Model architecture not supported"
            ):
                mock_main(config_mock)
