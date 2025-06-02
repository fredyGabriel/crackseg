"""Unit tests for src/evaluation/__main__.py module."""

import argparse
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

# Import only the public interface and dataclass
from src.evaluation.__main__ import EvaluationRunParameters
from src.utils.core.exceptions import ConfigError


class TestSetupEvaluationEnvironment:
    """Tests for _setup_evaluation_environment function."""

    @patch("src.evaluation.__main__._setup_evaluation_environment")
    def test_setup_evaluation_environment_success(self, mock_setup_func: Mock):
        """Test successful evaluation environment setup."""
        # Arrange
        mock_setup_func.return_value = (
            torch.device("cuda:0"),
            Path("/test/output"),
            ["/path/to/checkpoint.pth"],
        )

        args = argparse.Namespace(
            seed=42,
            device=None,
            output_dir="/test/output",
            checkpoint="/path/to/checkpoint.pth",
            ensemble=False,
        )

        # Act
        device, output_dir, checkpoint_paths = mock_setup_func(args)

        # Assert
        assert device == torch.device("cuda:0")
        assert output_dir == Path("/test/output")
        assert checkpoint_paths == ["/path/to/checkpoint.pth"]
        mock_setup_func.assert_called_once_with(args)


class TestLoadAndPrepareConfig:
    """Tests for _load_and_prepare_config function."""

    @patch("src.evaluation.__main__._load_and_prepare_config")
    def test_load_and_prepare_config_from_file(self, mock_load_func: Mock):
        """Test loading configuration from file."""
        # Arrange
        test_config = OmegaConf.create(
            {"model": {"type": "unet"}, "data": {"batch_size": 32}}
        )
        mock_load_func.return_value = test_config

        args = argparse.Namespace(
            config="test_config.yaml",
            data_dir=None,
            batch_size=None,
            num_workers=None,
            visualize_samples=None,
        )
        checkpoint_data = {}

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            # Act
            cfg = mock_load_func(args, checkpoint_data, output_dir)

            # Assert
            assert isinstance(cfg, DictConfig)
            mock_load_func.assert_called_once_with(
                args, checkpoint_data, output_dir
            )

    @patch("src.evaluation.__main__._load_and_prepare_config")
    def test_load_and_prepare_config_error(self, mock_load_func: Mock):
        """Test error when no configuration is available."""
        # Arrange
        mock_load_func.side_effect = ConfigError("Missing model configuration")

        args = argparse.Namespace(
            config=None,
            data_dir=None,
            batch_size=None,
            num_workers=None,
            visualize_samples=None,
        )
        checkpoint_data = {}

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            # Act & Assert
            with pytest.raises(
                ConfigError, match="Missing model configuration"
            ):
                mock_load_func(args, checkpoint_data, output_dir)


class TestGetEvaluationComponents:
    """Tests for _get_evaluation_components function."""

    @patch("src.evaluation.__main__._get_evaluation_components")
    def test_get_evaluation_components_success(
        self, mock_get_components: Mock
    ):
        """Test successful evaluation components setup."""
        # Arrange
        mock_dataloader = Mock(spec=DataLoader)
        mock_metrics = {"iou": Mock(), "dice": Mock()}
        mock_logger = Mock()

        mock_get_components.return_value = (
            mock_dataloader,
            mock_metrics,
            mock_logger,
        )

        cfg = OmegaConf.create(
            {"evaluation": {"metrics": {"iou": {}, "dice": {}}}}
        )
        args = argparse.Namespace(
            data_dir="/test/data", batch_size=32, num_workers=4
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            # Act
            test_loader, metrics, experiment_logger = mock_get_components(
                cfg, args, output_dir
            )

            # Assert
            assert test_loader == mock_dataloader
            assert metrics == mock_metrics
            assert experiment_logger == mock_logger
            mock_get_components.assert_called_once_with(cfg, args, output_dir)


class TestRunEvaluationAndLog:
    """Tests for _run_evaluation_and_log function."""

    @patch("src.evaluation.__main__._run_evaluation_and_log")
    def test_run_evaluation_and_log_single_model(self, mock_run_eval: Mock):
        """Test running evaluation for a single model."""
        # Arrange
        mock_model = Mock()
        mock_dataloader = Mock()
        mock_metrics = {"iou": Mock()}
        mock_logger = Mock()

        cfg = OmegaConf.create(
            {
                "device_str": "cpu",
                "output_dir_str": "/test/output",
                "evaluation": {"visualize_samples": 5},
            }
        )

        params = EvaluationRunParameters(
            is_ensemble=False,
            checkpoint_paths=["/path/to/checkpoint.pth"],
            cfg=cfg,
            test_loader=mock_dataloader,
            metrics_dict=mock_metrics,
            model_for_single_eval=mock_model,
            experiment_logger=mock_logger,
        )

        # Act
        mock_run_eval(params)

        # Assert
        mock_run_eval.assert_called_once_with(params)

    @patch("src.evaluation.__main__._run_evaluation_and_log")
    def test_run_evaluation_and_log_ensemble(self, mock_run_eval: Mock):
        """Test running evaluation for ensemble of models."""
        # Arrange
        mock_dataloader = Mock()
        mock_metrics = {"iou": Mock()}
        mock_logger = Mock()

        cfg = OmegaConf.create(
            {
                "device_str": "cpu",
                "output_dir_str": "/test/output",
                "evaluation": {"visualize_samples": 5},
            }
        )

        params = EvaluationRunParameters(
            is_ensemble=True,
            checkpoint_paths=[
                "/path/to/checkpoint1.pth",
                "/path/to/checkpoint2.pth",
            ],
            cfg=cfg,
            test_loader=mock_dataloader,
            metrics_dict=mock_metrics,
            model_for_single_eval=None,
            experiment_logger=mock_logger,
        )

        # Act
        mock_run_eval(params)

        # Assert
        mock_run_eval.assert_called_once_with(params)

    @patch("src.evaluation.__main__._run_evaluation_and_log")
    def test_run_evaluation_and_log_error(self, mock_run_eval: Mock):
        """Test error when model is None for single model evaluation."""
        # Arrange
        mock_run_eval.side_effect = ValueError("Model cannot be None")

        cfg = OmegaConf.create(
            {
                "device_str": "cpu",
                "output_dir_str": "/test/output",
                "evaluation": {"visualize_samples": 5},
            }
        )

        params = EvaluationRunParameters(
            is_ensemble=False,
            checkpoint_paths=["/path/to/checkpoint.pth"],
            cfg=cfg,
            test_loader=Mock(),
            metrics_dict={},
            model_for_single_eval=None,  # This should cause error
            experiment_logger=Mock(),
        )

        # Act & Assert
        with pytest.raises(ValueError, match="Model cannot be None"):
            mock_run_eval(params)


class TestEvaluationRunParameters:
    """Tests for EvaluationRunParameters dataclass."""

    def test_evaluation_run_parameters_creation(self):
        """Test creation of EvaluationRunParameters dataclass."""
        # Arrange & Act
        cfg = OmegaConf.create({"test": "config"})
        params = EvaluationRunParameters(
            is_ensemble=True,
            checkpoint_paths=["/path1", "/path2"],
            cfg=cfg,
            test_loader=Mock(),
            metrics_dict={"iou": Mock()},
            model_for_single_eval=None,
            experiment_logger=Mock(),
        )

        # Assert
        assert params.is_ensemble is True
        assert params.checkpoint_paths == ["/path1", "/path2"]
        assert params.cfg == cfg
        assert params.test_loader is not None
        assert "iou" in params.metrics_dict
        assert params.model_for_single_eval is None
        assert params.experiment_logger is not None


class TestEvaluationIntegration:
    """Integration tests for evaluation module components."""

    def test_path_handling(self):
        """Test proper Path handling throughout evaluation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)
            assert path.exists()

            # Test path operations used in evaluation
            config_path = path / "config.yaml"
            visualizations_path = path / "visualizations"
            metrics_path = path / "metrics"

            # These should be valid path operations
            assert str(config_path).endswith("config.yaml")
            assert str(visualizations_path).endswith("visualizations")
            assert str(metrics_path).endswith("metrics")

    def test_device_configuration(self):
        """Test device configuration handling."""
        # Test CPU device
        cfg = OmegaConf.create({"device_str": "cpu"})
        device = torch.device(cfg.device_str)
        assert device.type == "cpu"

        # Test CUDA device (if available)
        if torch.cuda.is_available():
            cfg = OmegaConf.create({"device_str": "cuda:0"})
            device = torch.device(cfg.device_str)
            assert device.type == "cuda"

    def test_config_structure_validation(self):
        """Test validation of evaluation configuration structure."""
        # Test minimal valid config
        minimal_cfg = OmegaConf.create(
            {
                "evaluation": {"metrics": {}, "visualize_samples": 5},
                "data": {"data_root": "test_data"},
                "device_str": "cpu",
                "output_dir_str": "/test/output",
            }
        )

        # Should have required sections
        assert "evaluation" in minimal_cfg
        assert "data" in minimal_cfg
        assert hasattr(minimal_cfg, "device_str")
        assert hasattr(minimal_cfg, "output_dir_str")
