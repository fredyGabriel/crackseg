#!/usr/bin/env python3
"""Tests for MetricsManager unified metric logging system."""

import json
import logging
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from crackseg.utils.logging.metrics_manager import MetricsManager


@pytest.fixture
def temp_experiment_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_logger() -> logging.Logger:
    """Create a mock logger for testing."""
    logger = Mock(spec=logging.Logger)
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    return logger


@pytest.fixture
def test_config() -> DictConfig:
    """Create a test configuration."""
    return OmegaConf.create({"test": True, "model": {"name": "test_model"}})


@pytest.fixture
def metrics_manager(
    temp_experiment_dir: Path,
    mock_logger: logging.Logger,
    test_config: DictConfig,
) -> MetricsManager:
    """Create a MetricsManager instance for testing."""
    return MetricsManager(
        experiment_dir=temp_experiment_dir,
        logger=mock_logger,
        config=test_config,
    )


class TestMetricsManagerInitialization:
    """Test MetricsManager initialization and setup."""

    def test_initialization_creates_directories(
        self, temp_experiment_dir: Path, mock_logger: logging.Logger
    ) -> None:
        """Test that initialization creates required directories."""
        manager = MetricsManager(
            experiment_dir=temp_experiment_dir,
            logger=mock_logger,
        )

        assert manager.metrics_dir.exists()
        assert manager.metrics_dir.is_dir()
        assert manager.experiment_dir == temp_experiment_dir

    def test_initialization_creates_file_paths(
        self, metrics_manager: MetricsManager
    ) -> None:
        """Test that initialization sets up correct file paths."""
        assert (
            metrics_manager.training_metrics_file.name
            == "training_metrics.jsonl"
        )
        assert (
            metrics_manager.validation_metrics_file.name
            == "validation_metrics.jsonl"
        )
        assert metrics_manager.test_metrics_file.name == "test_metrics.jsonl"
        assert metrics_manager.summary_file.name == "summary.json"

    def test_initialization_creates_summary(
        self, metrics_manager: MetricsManager
    ) -> None:
        """Test that initialization creates summary structure."""
        # Access via public interface instead of protected attribute
        summary_file = metrics_manager.summary_file
        if summary_file.exists():
            with open(summary_file) as f:
                summary = json.load(f)
        else:
            # Use public method to check summary structure
            export_path = metrics_manager.export_metrics_summary()
            with open(export_path) as f:
                summary_export = json.load(f)
                summary = {
                    "experiment_start": summary_export["experiment_info"].get(
                        "start_time"
                    ),
                    "total_epochs": summary_export["experiment_info"][
                        "total_epochs"
                    ],
                    "best_metrics": summary_export["best_metrics"],
                    "metric_history": summary_export["epoch_summaries"],
                }

        assert "experiment_start" in summary or summary.get("experiment_start")
        assert "total_epochs" in summary
        assert "best_metrics" in summary
        assert "metric_history" in summary


class TestMetricLogging:
    """Test metric logging functionality."""

    def test_log_training_metrics_with_tensors(
        self, metrics_manager: MetricsManager
    ) -> None:
        """Test logging training metrics with tensor values."""
        metrics: dict[str, float | torch.Tensor] = {
            "loss": torch.tensor(0.5),
            "accuracy": 0.85,
            "iou": torch.tensor([0.7, 0.74]).mean(),  # Multi-element tensor
        }

        metrics_manager.log_training_metrics(1, 10, metrics, "train")

        # Check that file was created and contains data
        assert metrics_manager.training_metrics_file.exists()

        # Read and verify content
        with open(metrics_manager.training_metrics_file) as f:
            entry = json.loads(f.readline().strip())

        assert entry["epoch"] == 1
        assert entry["step"] == 10
        assert entry["phase"] == "train"
        assert entry["metrics"]["loss"] == 0.5
        assert entry["metrics"]["accuracy"] == 0.85
        assert entry["metrics"]["iou"] == 0.72

    def test_log_training_metrics_different_phases(
        self, metrics_manager: MetricsManager
    ) -> None:
        """Test logging metrics to different phase files."""
        metrics: dict[str, float | torch.Tensor] = {"loss": 0.3}

        metrics_manager.log_training_metrics(1, 1, metrics, "train")
        metrics_manager.log_training_metrics(1, 1, metrics, "val")
        metrics_manager.log_training_metrics(1, 1, metrics, "test")

        assert metrics_manager.training_metrics_file.exists()
        assert metrics_manager.validation_metrics_file.exists()
        assert metrics_manager.test_metrics_file.exists()

    def test_log_epoch_summary(self, metrics_manager: MetricsManager) -> None:
        """Test logging epoch summary."""
        train_metrics = {"loss": 0.4, "accuracy": 0.87}
        val_metrics = {"loss": 0.35, "accuracy": 0.89}

        metrics_manager.log_epoch_summary(
            epoch=1,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            learning_rate=0.001,
        )

        # Check summary file was created
        assert metrics_manager.summary_file.exists()

        # Check summary content
        with open(metrics_manager.summary_file) as f:
            summary = json.load(f)

        assert summary["total_epochs"] == 1
        assert len(summary["metric_history"]) == 1

        history_entry = summary["metric_history"][0]
        assert history_entry["epoch"] == 1
        assert history_entry["train_metrics"] == train_metrics
        assert history_entry["val_metrics"] == val_metrics
        assert history_entry["learning_rate"] == 0.001

    def test_best_metrics_tracking(
        self, metrics_manager: MetricsManager
    ) -> None:
        """Test best metrics tracking logic."""
        # First epoch
        val_metrics1 = {"loss": 0.5, "accuracy": 0.8, "iou": 0.7}
        metrics_manager.log_epoch_summary(1, val_metrics=val_metrics1)

        # Second epoch with better metrics
        val_metrics2 = {"loss": 0.3, "accuracy": 0.9, "iou": 0.75}
        metrics_manager.log_epoch_summary(2, val_metrics=val_metrics2)

        # Third epoch with mixed results
        val_metrics3 = {"loss": 0.4, "accuracy": 0.85, "iou": 0.8}
        metrics_manager.log_epoch_summary(3, val_metrics=val_metrics3)

        # Check best metrics via public interface
        export_path = metrics_manager.export_metrics_summary()
        with open(export_path) as f:
            summary = json.load(f)

        best_metrics = summary["best_metrics"]

        # Loss should be best from epoch 2 (lowest)
        assert best_metrics["loss"]["value"] == 0.3
        assert best_metrics["loss"]["epoch"] == 2

        # Accuracy should be best from epoch 2 (highest)
        assert best_metrics["accuracy"]["value"] == 0.9
        assert best_metrics["accuracy"]["epoch"] == 2

        # IoU should be best from epoch 3 (highest)
        assert best_metrics["iou"]["value"] == 0.8
        assert best_metrics["iou"]["epoch"] == 3


class TestMetricRetrieval:
    """Test metric retrieval functionality."""

    def test_get_metric_history(self, metrics_manager: MetricsManager) -> None:
        """Test retrieving metric history."""
        # Log some metrics
        for epoch in range(1, 4):
            metrics: dict[str, float | torch.Tensor] = {
                "loss": 0.5 - epoch * 0.1
            }
            metrics_manager.log_training_metrics(epoch, 1, metrics, "val")

        history = metrics_manager.get_metric_history("loss", "val")

        assert len(history) == 3
        assert history[0] == (1, 0.4)
        assert history[1] == (2, 0.3)
        assert history[2] == (3, 0.2)

    def test_get_best_metric(self, metrics_manager: MetricsManager) -> None:
        """Test retrieving best metric."""
        val_metrics = {"accuracy": 0.85}
        metrics_manager.log_epoch_summary(1, val_metrics=val_metrics)

        best_accuracy = metrics_manager.get_best_metric("accuracy")
        assert best_accuracy is not None
        assert best_accuracy["value"] == 0.85
        assert best_accuracy["epoch"] == 1

    def test_get_nonexistent_metric(
        self, metrics_manager: MetricsManager
    ) -> None:
        """Test retrieving nonexistent metric."""
        best_metric = metrics_manager.get_best_metric("nonexistent")
        assert best_metric is None

    def test_get_metric_history_empty(
        self, metrics_manager: MetricsManager
    ) -> None:
        """Test retrieving history for empty metric."""
        history = metrics_manager.get_metric_history("nonexistent", "val")
        assert history == []


class TestMetricFormatting:
    """Test metric formatting and processing."""

    def test_extract_scalar_metrics_mixed_types(
        self, metrics_manager: MetricsManager
    ) -> None:
        """Test extracting scalar metrics from mixed types."""
        # Create metrics with proper types for the public interface
        mixed_metrics: dict[str, float | torch.Tensor] = {
            "tensor_scalar": torch.tensor(0.5),
            "tensor_multi": torch.tensor([0.6, 0.8]),
            "float_val": 0.7,
            "int_val": 1.0,  # Use float instead of int for consistency
        }

        # Test via public interface by logging and retrieving
        metrics_manager.log_training_metrics(1, 1, mixed_metrics, "train")

        # Verify the metrics were processed correctly by reading the file
        with open(metrics_manager.training_metrics_file) as f:
            entry = json.loads(f.readline().strip())

        scalar_metrics = entry["metrics"]
        assert scalar_metrics["tensor_scalar"] == 0.5
        assert scalar_metrics["tensor_multi"] == 0.7  # Mean of [0.6, 0.8]
        assert scalar_metrics["float_val"] == 0.7
        assert scalar_metrics["int_val"] == 1.0

    def test_format_metrics_for_console(
        self, metrics_manager: MetricsManager
    ) -> None:
        """Test metric formatting for console output via public interface."""
        metrics: dict[str, float | torch.Tensor] = {
            "loss": 0.1234,
            "accuracy": 0.8765,
        }

        # Test formatting via logging (which uses the internal formatter)
        metrics_manager.log_training_metrics(1, 1, metrics, "train")

        # Mock should have been called with formatted string
        mock_logger = cast(Mock, metrics_manager.logger)
        mock_logger.info.assert_called()

        # Verify the call contains formatted metrics
        call_args = mock_logger.info.call_args[0][0]
        assert "loss: 0.1234" in call_args
        assert "accuracy: 0.8765" in call_args


class TestExportFunctionality:
    """Test export and summary functionality."""

    def test_export_metrics_summary(
        self, metrics_manager: MetricsManager
    ) -> None:
        """Test exporting comprehensive metrics summary."""
        # Add some data
        train_metrics = {"loss": 0.4}
        val_metrics = {"loss": 0.35, "accuracy": 0.9}
        metrics_manager.log_epoch_summary(
            1, train_metrics=train_metrics, val_metrics=val_metrics
        )

        # Export summary
        export_path = metrics_manager.export_metrics_summary()

        assert export_path.exists()
        with open(export_path) as f:
            summary = json.load(f)

        assert "experiment_info" in summary
        assert "best_metrics" in summary
        assert "epoch_summaries" in summary
        assert "available_metrics" in summary

        assert summary["experiment_info"]["total_epochs"] == 1
        assert len(summary["epoch_summaries"]) == 1

    def test_export_metrics_summary_custom_path(
        self,
        metrics_manager: MetricsManager,
        temp_experiment_dir: Path,
    ) -> None:
        """Test exporting metrics summary to custom path."""
        custom_path = temp_experiment_dir / "custom_summary.json"
        export_path = metrics_manager.export_metrics_summary(custom_path)

        assert export_path == custom_path
        assert custom_path.exists()


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_phase_defaults_to_train(
        self, metrics_manager: MetricsManager
    ) -> None:
        """Test that invalid phase defaults to training file."""
        metrics: dict[str, float | torch.Tensor] = {"loss": 0.5}
        metrics_manager.log_training_metrics(1, 1, metrics, "invalid_phase")

        assert metrics_manager.training_metrics_file.exists()

    def test_metric_history_with_corrupted_file(
        self, metrics_manager: MetricsManager, temp_experiment_dir: Path
    ) -> None:
        """Test metric history retrieval with corrupted file."""
        # Create corrupted file
        with open(metrics_manager.validation_metrics_file, "w") as f:
            f.write("invalid json\n")

        history = metrics_manager.get_metric_history("loss", "val")
        assert history == []

    def test_logging_without_logger(self, temp_experiment_dir: Path) -> None:
        """Test MetricsManager works without logger."""
        manager = MetricsManager(
            experiment_dir=temp_experiment_dir,
            logger=None,
        )

        # Should not raise error
        metrics: dict[str, float | torch.Tensor] = {"loss": 0.5}
        manager.log_training_metrics(1, 1, metrics, "train")

        assert manager.training_metrics_file.exists()
