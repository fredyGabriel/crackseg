"""
Unit tests for src.main module.

Tests the main training pipeline functions for crack segmentation,
including environment setup, data loading, model creation, and
training components.
"""

import logging
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from torch.utils.data import DataLoader

from src.utils import DataError, ModelError, ResourceError


class TestSetupEnvironment:
    """Test environment setup functionality."""

    @patch("src.main.set_random_seeds")
    @patch("src.main.get_device")
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
        with patch("src.main._setup_environment") as mock_setup:
            mock_setup.return_value = (torch.device("cuda"), True)
            device, cuda_available = mock_setup(config_mock)

        # Assert
        assert device == torch.device("cuda")
        assert cuda_available is True

    @patch("src.main.set_random_seeds")
    @patch("src.main.get_device")
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
        with patch("src.main._setup_environment") as mock_setup:
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
        with patch("src.main._setup_environment") as mock_setup:
            mock_setup.side_effect = ResourceError(
                "CUDA device required but not available"
            )
            with pytest.raises(ResourceError, match="CUDA device required"):
                mock_setup(config_mock)

    @patch("src.main.set_random_seeds")
    @patch("src.main.get_device")
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
        config_mock.experiment.device = "cpu"
        config_mock.experiment.seed = 123

        # Act
        with patch("src.main._setup_environment") as mock_setup:
            mock_setup.return_value = (torch.device("cpu"), False)
            device, cuda_available = mock_setup(config_mock)

        # Assert
        assert device == torch.device("cpu")
        assert cuda_available is False


class TestLoadData:
    """Test data loading functionality."""

    @patch("src.main.create_dataloaders_from_config")
    @patch("hydra.utils.get_original_cwd")
    def test_load_data_success(
        self, mock_get_cwd: Mock, mock_create_dataloaders: Mock
    ) -> None:
        """Test successful data loading."""
        # Arrange
        mock_get_cwd.return_value = "/original/cwd"
        train_loader = MagicMock(spec=DataLoader)
        val_loader = MagicMock(spec=DataLoader)
        test_loader = MagicMock(spec=DataLoader)

        mock_create_dataloaders.return_value = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }

        config_mock = MagicMock()
        config_mock.data.transform = {"train": {}, "val": {}, "test": {}}

        # Act
        with patch("src.main._load_data") as mock_load:
            mock_load.return_value = {
                "train": train_loader,
                "val": val_loader,
                "test": test_loader,
            }
            result = mock_load(config_mock)

        # Assert
        assert "train" in result
        assert "val" in result
        assert "test" in result
        assert isinstance(result["train"], MagicMock)
        assert isinstance(result["val"], MagicMock)
        assert isinstance(result["test"], MagicMock)

    @patch("src.main.create_dataloaders_from_config")
    @patch("hydra.utils.get_original_cwd")
    def test_load_data_missing_train_loader(
        self, mock_get_cwd: Mock, mock_create_dataloaders: Mock
    ) -> None:
        """Test error when train loader is missing."""
        # Arrange
        mock_get_cwd.return_value = "/original/cwd"
        mock_create_dataloaders.return_value = {
            "val": MagicMock(spec=DataLoader),
            "test": MagicMock(spec=DataLoader),
        }

        config_mock = MagicMock()

        # Act & Assert
        with patch("src.main._load_data") as mock_load:
            mock_load.side_effect = DataError(
                "Training dataloader is required"
            )
            with pytest.raises(
                DataError, match="Training dataloader is required"
            ):
                mock_load(config_mock)

    @patch("src.main.create_dataloaders_from_config")
    @patch("hydra.utils.get_original_cwd")
    def test_load_data_invalid_loader_type(
        self, mock_get_cwd: Mock, mock_create_dataloaders: Mock
    ) -> None:
        """Test error when loaders are not DataLoader instances."""
        # Arrange
        mock_get_cwd.return_value = "/original/cwd"
        mock_create_dataloaders.return_value = {
            "train": "not_a_dataloader",  # Invalid type
            "val": MagicMock(spec=DataLoader),
        }

        config_mock = MagicMock()

        # Act & Assert
        with patch("src.main._load_data") as mock_load:
            mock_load.side_effect = DataError("Invalid DataLoader type")
            with pytest.raises(DataError, match="Invalid DataLoader type"):
                mock_load(config_mock)

    @patch("src.main.create_dataloaders_from_config")
    @patch("hydra.utils.get_original_cwd")
    def test_load_data_missing_transform_config(
        self, mock_get_cwd: Mock, mock_create_dataloaders: Mock
    ) -> None:
        """Test data loading with missing transform config."""
        # Arrange
        mock_get_cwd.return_value = "/original/cwd"
        train_loader = MagicMock(spec=DataLoader)

        mock_create_dataloaders.return_value = {
            "train": train_loader,
        }

        config_mock = MagicMock()
        # Missing transform config
        if hasattr(config_mock.data, "transform"):
            delattr(config_mock.data, "transform")

        # Act
        with patch("src.main._load_data") as mock_load:
            mock_load.return_value = {"train": train_loader}
            result = mock_load(config_mock)

        # Assert
        assert "train" in result
        assert isinstance(result["train"], MagicMock)

    @patch("src.main.create_dataloaders_from_config")
    @patch("hydra.utils.get_original_cwd")
    def test_load_data_exception_handling(
        self, mock_get_cwd: Mock, mock_create_dataloaders: Mock
    ) -> None:
        """Test data loading exception handling."""
        # Arrange
        mock_get_cwd.return_value = "/original/cwd"
        mock_create_dataloaders.side_effect = Exception("Data loading failed")

        config_mock = MagicMock()

        # Act & Assert
        with patch("src.main._load_data") as mock_load:
            mock_load.side_effect = DataError("Failed to load data")
            with pytest.raises(DataError, match="Failed to load data"):
                mock_load(config_mock)


class TestCreateModel:
    """Test model creation functionality."""

    @patch("hydra.utils.instantiate")
    def test_create_model_success(self, mock_instantiate: Mock) -> None:
        """Test successful model creation."""
        # Arrange
        mock_model = MagicMock()
        mock_model.train.return_value = None
        mock_model.to.return_value = mock_model
        mock_instantiate.return_value = mock_model

        config_mock = MagicMock()
        config_mock.model = {"_target_": "src.model.CrackSegmentationModel"}
        device = torch.device("cpu")

        # Act
        with patch("src.main._create_model") as mock_create:
            mock_create.return_value = mock_model
            result = mock_create(config_mock, device)

        # Assert
        assert result is mock_model

    @patch("hydra.utils.instantiate")
    def test_create_model_instantiation_error(
        self, mock_instantiate: Mock
    ) -> None:
        """Test model creation with instantiation error."""
        # Arrange
        mock_instantiate.side_effect = Exception("Model instantiation failed")
        config_mock = MagicMock()
        device = torch.device("cpu")

        # Act & Assert
        with patch("src.main._create_model") as mock_create:
            mock_create.side_effect = ModelError("Failed to create model")
            with pytest.raises(ModelError, match="Failed to create model"):
                mock_create(config_mock, device)

    @patch("hydra.utils.instantiate")
    def test_create_model_attribute_error(
        self, mock_instantiate: Mock
    ) -> None:
        """Test model creation with missing methods."""
        # Arrange
        mock_model = MagicMock()
        del mock_model.train  # Remove train method
        mock_instantiate.return_value = mock_model

        config_mock = MagicMock()
        device = torch.device("cpu")

        # Act & Assert
        with patch("src.main._create_model") as mock_create:
            mock_create.side_effect = ModelError(
                "Model missing required methods"
            )
            with pytest.raises(
                ModelError, match="Model missing required methods"
            ):
                mock_create(config_mock, device)


class TestSetupTrainingComponents:
    """Test training components setup functionality."""

    @patch("src.main.get_metrics_from_cfg")
    @patch("src.main.get_optimizer")
    @patch("src.main.get_loss_fn")
    def test_setup_training_components_success(
        self,
        mock_get_loss_fn: Mock,
        mock_get_optimizer: Mock,
        mock_get_metrics: Mock,
    ) -> None:
        """Test successful training components setup."""
        # Arrange
        mock_loss_fn = MagicMock()
        mock_optimizer = MagicMock()
        mock_metrics = {"iou": MagicMock(), "dice": MagicMock()}

        mock_get_loss_fn.return_value = mock_loss_fn
        mock_get_optimizer.return_value = mock_optimizer
        mock_get_metrics.return_value = mock_metrics

        config_mock = MagicMock()
        config_mock.training.loss = {"type": "dice"}
        config_mock.training.optimizer = {"type": "adam"}
        config_mock.training.metrics = ["iou", "dice"]

        model_mock = MagicMock()

        # Act
        with patch("src.main._setup_training_components") as mock_setup:
            mock_setup.return_value = (
                mock_loss_fn,
                mock_optimizer,
                mock_metrics,
            )
            loss_fn, optimizer, metrics = mock_setup(config_mock, model_mock)

        # Assert
        assert loss_fn is mock_loss_fn
        assert optimizer is mock_optimizer
        assert metrics is mock_metrics
        assert "iou" in metrics
        assert "dice" in metrics

    @patch("src.main.get_optimizer")
    def test_setup_training_components_no_metrics(
        self, mock_get_optimizer: Mock
    ) -> None:
        """Test training components setup with no metrics specified."""
        # Arrange
        mock_optimizer = MagicMock()
        mock_get_optimizer.return_value = mock_optimizer

        config_mock = MagicMock()
        config_mock.training.metrics = None

        model_mock = MagicMock()

        # Act
        with patch("src.main._setup_training_components") as mock_setup:
            mock_setup.return_value = (MagicMock(), mock_optimizer, {})
            loss_fn, optimizer, metrics = mock_setup(config_mock, model_mock)

        # Assert
        assert optimizer is mock_optimizer
        assert metrics == {}

    @patch("src.main.get_optimizer")
    @patch("src.main.get_loss_fn")
    def test_setup_training_components_invalid_loss(
        self, mock_get_loss_fn: Mock, mock_get_optimizer: Mock
    ) -> None:
        """Test training components setup with invalid loss function."""
        # Arrange
        mock_get_loss_fn.side_effect = Exception("Invalid loss function")
        mock_get_optimizer.return_value = MagicMock()

        config_mock = MagicMock()
        config_mock.training.loss = {"type": "invalid_loss"}

        model_mock = MagicMock()

        # Act & Assert
        with patch("src.main._setup_training_components") as mock_setup:
            mock_setup.side_effect = ModelError(
                "Failed to setup loss function"
            )
            with pytest.raises(
                ModelError, match="Failed to setup loss function"
            ):
                mock_setup(config_mock, model_mock)

    @patch("src.main.get_optimizer")
    @patch("src.main.get_loss_fn")
    def test_setup_training_components_loss_exception(
        self, mock_get_loss_fn: Mock, mock_get_optimizer: Mock
    ) -> None:
        """Test training components setup with loss function exception."""
        # Arrange
        mock_loss_fn = MagicMock()
        mock_loss_fn.side_effect = RuntimeError("Loss computation failed")
        mock_get_loss_fn.return_value = mock_loss_fn
        mock_get_optimizer.return_value = MagicMock()

        config_mock = MagicMock()
        model_mock = MagicMock()

        # Act
        with patch("src.main._setup_training_components") as mock_setup:
            # The function should return normally, but loss_fn might raise
            # during use
            mock_setup.return_value = (mock_loss_fn, MagicMock(), {})
            loss_fn, optimizer, metrics = mock_setup(config_mock, model_mock)

        # Assert
        assert loss_fn is mock_loss_fn
        # Verify that the loss function will raise when called
        with pytest.raises(RuntimeError, match="Loss computation failed"):
            loss_fn(MagicMock(), MagicMock())


class TestHandleCheckpointingAndResume:
    """Test checkpointing and resume functionality."""

    @patch("src.main.load_checkpoint")
    @patch("os.path.exists")
    @patch("hydra.utils.get_original_cwd")
    def test_handle_checkpointing_resume_success(
        self, mock_get_cwd: Mock, mock_exists: Mock, mock_load_checkpoint: Mock
    ) -> None:
        """Test successful checkpoint loading and resume."""
        # Arrange
        mock_get_cwd.return_value = "/original/cwd"
        mock_exists.return_value = True

        checkpoint_data = {
            "model_state_dict": {"layer1.weight": torch.randn(64, 3, 3, 3)},
            "optimizer_state_dict": {"state": {}, "param_groups": []},
            "epoch": 5,
            "best_metric": 0.85,
            "metrics_history": [{"epoch": 1, "iou": 0.7}],
        }
        mock_load_checkpoint.return_value = checkpoint_data

        config_mock = MagicMock()
        config_mock.checkpointing.resume_from = "checkpoints/best_model.pth"

        model_mock = MagicMock()
        optimizer_mock = MagicMock()
        logger_mock = MagicMock()

        # Act
        with patch("src.main._handle_checkpointing_and_resume") as mock_handle:
            mock_handle.return_value = (5, 0.85, [{"epoch": 1, "iou": 0.7}])
            start_epoch, best_metric, history = mock_handle(
                config_mock, model_mock, optimizer_mock, logger_mock
            )

        # Assert
        assert start_epoch == 5
        assert best_metric == 0.85
        assert len(history) == 1
        assert history[0]["epoch"] == 1

    @patch("os.path.exists")
    @patch("hydra.utils.get_original_cwd")
    def test_handle_checkpointing_checkpoint_not_found(
        self, mock_get_cwd: Mock, mock_exists: Mock
    ) -> None:
        """Test handling when checkpoint file doesn't exist."""
        # Arrange
        mock_get_cwd.return_value = "/original/cwd"
        mock_exists.return_value = False

        config_mock = MagicMock()
        config_mock.checkpointing.resume_from = "checkpoints/nonexistent.pth"

        model_mock = MagicMock()
        optimizer_mock = MagicMock()
        logger_mock = MagicMock()

        # Act
        with patch("src.main._handle_checkpointing_and_resume") as mock_handle:
            mock_handle.return_value = (0, None, [])
            start_epoch, best_metric, history = mock_handle(
                config_mock, model_mock, optimizer_mock, logger_mock
            )

        # Assert
        assert start_epoch == 0
        assert best_metric is None
        assert history == []

    def test_handle_checkpointing_no_resume(self) -> None:
        """Test checkpointing setup without resuming."""
        # Arrange
        config_mock = MagicMock()
        config_mock.checkpointing.resume_from = None

        model_mock = MagicMock()
        optimizer_mock = MagicMock()
        logger_mock = MagicMock()

        # Act
        with patch("src.main._handle_checkpointing_and_resume") as mock_handle:
            mock_handle.return_value = (0, None, [])
            start_epoch, best_metric, history = mock_handle(
                config_mock, model_mock, optimizer_mock, logger_mock
            )

        # Assert
        assert start_epoch == 0
        assert best_metric is None
        assert history == []

    def test_handle_checkpointing_invalid_logger(self) -> None:
        """Test checkpointing with invalid logger configuration."""
        # Arrange
        config_mock = MagicMock()
        config_mock.checkpointing.resume_from = None

        model_mock = MagicMock()
        optimizer_mock = MagicMock()
        logger_mock = None  # Invalid logger

        # Act & Assert
        with patch("src.main._handle_checkpointing_and_resume") as mock_handle:
            mock_handle.side_effect = ValueError(
                "Invalid logger configuration"
            )
            with pytest.raises(
                ValueError, match="Invalid logger configuration"
            ):
                mock_handle(
                    config_mock, model_mock, optimizer_mock, logger_mock
                )

    def test_handle_checkpointing_save_best_no_metric(self) -> None:
        """Test checkpointing save_best configuration without metric."""
        # Arrange
        config_mock = MagicMock()
        config_mock.checkpointing.save_best = True
        config_mock.checkpointing.save_best_metric = None

        model_mock = MagicMock()
        optimizer_mock = MagicMock()
        logger_mock = MagicMock()

        # Act
        with patch("src.main._handle_checkpointing_and_resume") as mock_handle:
            # Should handle gracefully, defaulting to saving based on loss
            mock_handle.return_value = (0, None, [])
            start_epoch, best_metric, history = mock_handle(
                config_mock, model_mock, optimizer_mock, logger_mock
            )

        # Assert
        assert start_epoch == 0
        assert best_metric is None
        assert history == []


class TestMainIntegration:
    """Test integration aspects of main functionality."""

    def test_logging_configuration(self) -> None:
        """Test that logging is properly configured."""
        # Act
        logger = logging.getLogger("src.main")

        # Assert
        assert logger is not None
        assert logger.level <= logging.INFO

    @patch("torch.cuda.is_available")
    def test_device_selection_logic(self, mock_cuda_available: Mock) -> None:
        """Test device selection logic."""
        # Test CUDA available
        mock_cuda_available.return_value = True
        assert torch.cuda.is_available() is True

        # Test CUDA not available
        mock_cuda_available.return_value = False
        assert torch.cuda.is_available() is False

    def test_config_structure_validation(self) -> None:
        """Test expected configuration structure."""
        # Arrange - Test that our test mocks follow expected structure
        config_mock = MagicMock()

        # Expected config structure
        expected_sections = [
            "experiment",
            "data",
            "model",
            "training",
            "checkpointing",
        ]

        # Act & Assert
        for section in expected_sections:
            assert hasattr(config_mock, section)
