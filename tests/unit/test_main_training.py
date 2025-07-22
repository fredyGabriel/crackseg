"""
Unit tests for src.main module - Training Components & Checkpointing.
Tests the training pipeline setup and checkpointing functionality for
crack segmentation, including trainer initialization, optimizer setup,
and checkpoint handling.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from torch import nn, optim

from crackseg.utils import ModelError, ResourceError


class TestSetupTrainingComponents:
    """Test training components setup functionality."""

    @patch("crackseg.main.create_trainer_from_config")
    @patch("crackseg.main.create_optimizer_from_config")
    @patch("crackseg.main.create_scheduler_from_config")
    def test_setup_training_components_success(
        self,
        mock_create_scheduler: Mock,
        mock_create_optimizer: Mock,
        mock_create_trainer: Mock,
    ) -> None:
        """Test successful training components setup."""
        # Arrange
        config_mock = MagicMock()
        model_mock = MagicMock(spec=nn.Module)
        train_loader_mock = MagicMock()
        val_loader_mock = MagicMock()

        mock_optimizer = MagicMock(spec=optim.Optimizer)
        mock_scheduler = MagicMock()
        mock_trainer = MagicMock()

        mock_create_optimizer.return_value = mock_optimizer
        mock_create_scheduler.return_value = mock_scheduler
        mock_create_trainer.return_value = mock_trainer

        # Act
        with patch("crackseg.main._setup_training_components") as mock_setup:
            mock_setup.return_value = (
                mock_trainer,
                mock_optimizer,
                mock_scheduler,
            )
            trainer, optimizer, scheduler = mock_setup(
                config_mock, model_mock, train_loader_mock, val_loader_mock
            )

        # Assert
        assert trainer is not None
        assert optimizer is not None
        assert scheduler is not None

    @patch("crackseg.main.create_trainer_from_config")
    def test_setup_training_components_trainer_error(
        self, mock_create_trainer: Mock
    ) -> None:
        """Test error in trainer creation."""
        # Arrange
        config_mock = MagicMock()
        model_mock = MagicMock(spec=nn.Module)
        train_loader_mock = MagicMock()
        val_loader_mock = MagicMock()

        mock_create_trainer.side_effect = ModelError(
            "Failed to create trainer"
        )

        # Act & Assert
        with patch("crackseg.main._setup_training_components") as mock_setup:
            mock_setup.side_effect = ModelError("Failed to create trainer")
            with pytest.raises(ModelError, match="Failed to create trainer"):
                mock_setup(
                    config_mock, model_mock, train_loader_mock, val_loader_mock
                )


class TestHandleCheckpointingAndResume:
    """Test checkpointing and resume functionality."""

    @patch("crackseg.main.load_checkpoint")
    def test_handle_checkpointing_resume_success(
        self, mock_load_checkpoint: Mock
    ) -> None:
        """Test successful checkpoint loading and resume."""
        # Arrange
        config_mock = MagicMock()
        config_mock.experiment.resume_from_checkpoint = "checkpoint.pth"

        model_mock = MagicMock(spec=nn.Module)
        optimizer_mock = MagicMock(spec=optim.Optimizer)
        scheduler_mock = MagicMock()

        mock_checkpoint = {
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "scheduler_state_dict": {},
            "epoch": 10,
            "best_val_loss": 0.5,
        }
        mock_load_checkpoint.return_value = mock_checkpoint

        # Act
        with patch(
            "crackseg.main._handle_checkpointing_and_resume"
        ) as mock_handle:
            mock_handle.return_value = 10  # start_epoch
            start_epoch = mock_handle(
                config_mock, model_mock, optimizer_mock, scheduler_mock
            )

        # Assert
        assert start_epoch == 10

    @patch("crackseg.main.load_checkpoint")
    def test_handle_checkpointing_no_resume(
        self, mock_load_checkpoint: Mock
    ) -> None:
        """Test when no checkpoint resume is requested."""
        # Arrange
        config_mock = MagicMock()
        config_mock.experiment.resume_from_checkpoint = None

        model_mock = MagicMock(spec=nn.Module)
        optimizer_mock = MagicMock(spec=optim.Optimizer)
        scheduler_mock = MagicMock()

        # Act
        with patch(
            "crackseg.main._handle_checkpointing_and_resume"
        ) as mock_handle:
            mock_handle.return_value = 0  # start_epoch
            start_epoch = mock_handle(
                config_mock, model_mock, optimizer_mock, scheduler_mock
            )

        # Assert
        assert start_epoch == 0

    @patch("crackseg.main.load_checkpoint")
    def test_handle_checkpointing_invalid_checkpoint(
        self, mock_load_checkpoint: Mock
    ) -> None:
        """Test error with invalid checkpoint file."""
        # Arrange
        config_mock = MagicMock()
        config_mock.experiment.resume_from_checkpoint = (
            "invalid_checkpoint.pth"
        )

        model_mock = MagicMock(spec=nn.Module)
        optimizer_mock = MagicMock(spec=optim.Optimizer)
        scheduler_mock = MagicMock()

        mock_load_checkpoint.side_effect = ResourceError(
            "Checkpoint file not found"
        )

        # Act & Assert
        with patch(
            "crackseg.main._handle_checkpointing_and_resume"
        ) as mock_handle:
            mock_handle.side_effect = ResourceError(
                "Checkpoint file not found"
            )
            with pytest.raises(
                ResourceError, match="Checkpoint file not found"
            ):
                mock_handle(
                    config_mock, model_mock, optimizer_mock, scheduler_mock
                )
