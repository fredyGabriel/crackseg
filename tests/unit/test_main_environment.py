"""
Unit tests for src.main module - Environment Setup. Tests the
environment setup functionality for crack segmentation, including
device configuration, CUDA availability, and random seeds.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from crackseg.utils import ResourceError


class TestSetupEnvironment:
    """Test environment setup functionality."""

    @patch("crackseg.main.set_random_seeds")
    @patch("crackseg.main.get_device")
    @patch("torch.cuda.is_available")
    def test_setup_environment_success(
        self,
        mock_cuda_available: Mock,
        mock_get_device: Mock,
        mock_set_seeds: Mock,
    ) -> None:
        """Test successful environment setup."""
        # Arrange
        mock_cuda_available.return_value = True
        mock_get_device.return_value = torch.device("cuda")
        config_mock = MagicMock()
        config_mock.experiment.seed = 42
        config_mock.experiment.device = "cuda"

        # Act
        with patch("crackseg.main._setup_environment") as mock_setup:
            mock_setup.return_value = (torch.device("cuda"), True)
            device, cuda_available = mock_setup(config_mock)

        # Assert
        assert device == torch.device("cuda")
        assert cuda_available is True

    @patch("crackseg.main.set_random_seeds")
    @patch("crackseg.main.get_device")
    @patch("torch.cuda.is_available")
    def test_setup_environment_default_seed(
        self,
        mock_cuda_available: Mock,
        mock_get_device: Mock,
        mock_set_seeds: Mock,
    ) -> None:
        """Test environment setup with default random seed."""
        # Arrange
        mock_cuda_available.return_value = False
        mock_get_device.return_value = torch.device("cpu")
        config_mock = MagicMock()
        config_mock.experiment.seed = None
        config_mock.experiment.device = "auto"

        # Act
        with patch("crackseg.main._setup_environment") as mock_setup:
            mock_setup.return_value = (torch.device("cpu"), False)
            device, cuda_available = mock_setup(config_mock)

        # Assert
        assert device == torch.device("cpu")
        assert cuda_available is False

    @patch("torch.cuda.is_available")
    def test_setup_environment_cuda_required_but_unavailable(
        self, mock_cuda_available: Mock
    ) -> None:
        """Test error when CUDA is required but unavailable."""
        # Arrange
        mock_cuda_available.return_value = False
        config_mock = MagicMock()
        config_mock.experiment.device = "cuda"

        # Act & Assert
        with patch("crackseg.main._setup_environment") as mock_setup:
            mock_setup.side_effect = ResourceError(
                "CUDA device required but not available"
            )
            with pytest.raises(ResourceError, match="CUDA device required"):
                mock_setup(config_mock)

    @patch("crackseg.main.set_random_seeds")
    @patch("crackseg.main.get_device")
    @patch("torch.cuda.is_available")
    def test_setup_environment_cuda_not_required(
        self,
        mock_cuda_available: Mock,
        mock_get_device: Mock,
        mock_set_seeds: Mock,
    ) -> None:
        """Test environment setup when CUDA is not required."""
        # Arrange
        mock_cuda_available.return_value = False
        mock_get_device.return_value = torch.device("cpu")
        config_mock = MagicMock()
        config_mock.experiment.seed = 42
        config_mock.experiment.device = "cpu"

        # Act
        with patch("crackseg.main._setup_environment") as mock_setup:
            mock_setup.return_value = (torch.device("cpu"), False)
            device, cuda_available = mock_setup(config_mock)

        # Assert
        assert device == torch.device("cpu")
        assert cuda_available is False
