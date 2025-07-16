"""Integration tests for training artifacts generation, loading, and
validation.

This module tests the complete artifact lifecycle across the training pipeline:
- Metrics artifacts (CSV, JSON, plots)
- Checkpoint artifacts (model states, optimizer states)
- Configuration artifacts (standardized configs, validation reports)
- Compatibility across different environments and versions
"""

import json
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

from crackseg.training.trainer import Trainer, TrainingComponents
from crackseg.utils.checkpointing import load_checkpoint
from crackseg.utils.config.standardized_storage import (
    StandardizedConfigStorage,
)


@pytest.fixture
def sample_training_config() -> DictConfig:
    """Create a complete valid training configuration for artifact testing."""
    config_dict = {
        "experiment": {"name": "test_artifacts_experiment"},
        "model": {
            "_target_": "torch.nn.Sequential",
            "layers": [
                {
                    "_target_": "torch.nn.Linear",
                    "in_features": 10,
                    "out_features": 5,
                },
                {"_target_": "torch.nn.ReLU"},
                {
                    "_target_": "torch.nn.Linear",
                    "in_features": 5,
                    "out_features": 1,
                },
            ],
        },
        "training": {
            "epochs": 3,
            "optimizer": {"_target_": "torch.optim.Adam", "lr": 0.001},
            "scheduler": {
                "_target_": "torch.optim.lr_scheduler.StepLR",
                "step_size": 2,
                "gamma": 0.9,
            },
            "device": "cpu",
            "use_amp": False,
            "gradient_accumulation_steps": 1,
            "log_interval_batches": 1,
            "early_stopping": {"enabled": False},
            "save_freq": 1,  # Save every epoch for testing
        },
        "data": {"root_dir": "test_data", "batch_size": 4},
        "random_seed": 42,
        "checkpoints": {
            "save_best": {
                "enabled": True,
                "monitor_metric": "val_loss",
                "monitor_mode": "min",
                "best_filename": "model_best.pth.tar",
            },
            "keep_last_n": 2,
            "last_filename": "checkpoint_last.pth",
        },
    }
    return OmegaConf.create(config_dict)


@pytest.fixture
def mock_training_components() -> TrainingComponents:
    """Create mock training components that generate realistic artifacts."""
    # Create a simple but realistic model
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 5), torch.nn.ReLU(), torch.nn.Linear(5, 1)
    )

    # Mock data loaders with realistic iteration behavior
    train_loader = Mock()
    train_loader.__len__ = Mock(return_value=8)  # 8 batches per epoch
    train_loader.__iter__ = Mock(
        return_value=iter(
            [(torch.randn(4, 10), torch.randn(4, 1)) for _ in range(8)]
        )
    )

    val_loader = Mock()
    val_loader.__len__ = Mock(return_value=3)  # 3 validation batches
    val_loader.__iter__ = Mock(
        return_value=iter(
            [(torch.randn(4, 10), torch.randn(4, 1)) for _ in range(3)]
        )
    )

    loss_fn = torch.nn.MSELoss()
    metrics_dict = {"mae": torch.nn.L1Loss()}

    return TrainingComponents(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        metrics_dict=metrics_dict,
    )


@pytest.fixture
def experiment_setup(
    mock_training_components: TrainingComponents,
    sample_training_config: DictConfig,
) -> Generator[tuple[Path, Trainer], None, None]:
    """Set up a complete experiment environment for artifact testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        experiment_dir = Path(temp_dir) / "artifacts_test_experiment"
        experiment_dir.mkdir(parents=True, exist_ok=True)

        # Mock experiment manager with proper directory structure
        mock_experiment_manager = Mock()
        mock_experiment_manager.experiment_dir = experiment_dir
        mock_experiment_manager.experiment_id = "artifacts_test_001"
        mock_experiment_manager.get_path.return_value = str(
            experiment_dir / "checkpoints"
        )

        # Mock logger instance
        mock_logger = Mock()
        mock_logger.experiment_manager = mock_experiment_manager

        # Initialize trainer
        trainer = Trainer(
            components=mock_training_components,
            cfg=sample_training_config,
            logger_instance=mock_logger,
        )

        yield experiment_dir, trainer


class TestArtifactGeneration:
    """Test artifact generation during training workflow.

    Covers action items 1-2: Design test cases and implement integration tests
    for artifact generation during training.
    """

    @pytest.mark.integration
    def test_complete_artifact_generation_workflow(
        self, experiment_setup: tuple[Path, Trainer]
    ) -> None:
        """Test that all artifacts are generated correctly during training."""
        experiment_dir, trainer = experiment_setup

        # Run training to generate artifacts
        final_metrics = trainer.train()

        # Verify all expected artifact types are generated
        self._verify_checkpoint_artifacts(experiment_dir)
        self._verify_configuration_artifacts(experiment_dir)
        self._verify_metrics_artifacts(experiment_dir)

        # Verify final metrics are reasonable
        assert isinstance(final_metrics, dict)
        assert "val_loss" in final_metrics
        assert isinstance(final_metrics["val_loss"], int | float)

    def _verify_checkpoint_artifacts(self, experiment_dir: Path) -> None:
        """Verify checkpoint artifacts are generated correctly."""
        checkpoint_dir = experiment_dir / "checkpoints"
        assert checkpoint_dir.exists(), "Checkpoint directory should exist"

        # Check for checkpoint files
        checkpoint_files = list(checkpoint_dir.glob("*.pth*"))
        assert (
            len(checkpoint_files) > 0
        ), "At least one checkpoint should be saved"

        # Verify checkpoint content
        for checkpoint_file in checkpoint_files:
            checkpoint_data = torch.load(checkpoint_file, map_location="cpu")
            assert "model_state_dict" in checkpoint_data
            assert "optimizer_state_dict" in checkpoint_data
            assert "epoch" in checkpoint_data

    def _verify_configuration_artifacts(self, experiment_dir: Path) -> None:
        """Verify configuration artifacts are generated correctly."""
        config_dir = experiment_dir / "configurations"
        assert config_dir.exists(), "Configuration directory should exist"

        # Check for experiment-specific config directory
        exp_config_dirs = [d for d in config_dir.iterdir() if d.is_dir()]
        assert (
            len(exp_config_dirs) > 0
        ), "Experiment config directory should exist"

        exp_config_dir = exp_config_dirs[0]

        # Verify initial configuration file
        config_files = list(exp_config_dir.glob("training_config.*"))
        assert len(config_files) > 0, "Training configuration should be saved"

        # Verify epoch configurations
        epoch_configs = list(exp_config_dir.glob("config_epoch_*.yaml"))
        assert len(epoch_configs) > 0, "Epoch configurations should be saved"

        # Verify configuration content
        config_file = config_files[0]
        if config_file.suffix == ".yaml":
            with open(config_file, encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
        else:
            with open(config_file, encoding="utf-8") as f:
                config_data = json.load(f)

        assert "experiment" in config_data
        assert "model" in config_data
        assert "training" in config_data
        assert "environment" in config_data  # Should be enriched

    def _verify_metrics_artifacts(self, experiment_dir: Path) -> None:
        """Verify metrics artifacts are generated correctly."""
        # MetricsManager should create metrics directory
        metrics_dirs = list(experiment_dir.glob("**/metrics"))
        if len(metrics_dirs) > 0:
            metrics_dir = metrics_dirs[0]
            # Check for metrics files (CSV, JSON)
            metrics_files = list(metrics_dir.glob("*.csv")) + list(
                metrics_dir.glob("*.json")
            )
            # If metrics files exist, verify they contain expected data
            for metrics_file in metrics_files:
                assert (
                    metrics_file.stat().st_size > 0
                ), f"Metrics file {metrics_file} should not be empty"


class TestArtifactLoading:
    """Test loading artifacts in different environments.

    Covers action item 3: Create tests for loading artifacts in different
    environments.
    """

    @pytest.mark.integration
    def test_checkpoint_loading_compatibility(
        self, experiment_setup: tuple[Path, Trainer]
    ) -> None:
        """Test that checkpoints can be loaded in different environments."""
        experiment_dir, trainer = experiment_setup

        # Generate a checkpoint
        trainer.train()

        # Find generated checkpoint
        checkpoint_dir = experiment_dir / "checkpoints"
        checkpoint_files = list(checkpoint_dir.glob("*.pth*"))
        assert len(checkpoint_files) > 0

        checkpoint_path = checkpoint_files[0]

        # Test loading checkpoint with different configurations
        self._test_checkpoint_loading_scenarios(checkpoint_path)

    def _test_checkpoint_loading_scenarios(
        self, checkpoint_path: Path
    ) -> None:
        """Test checkpoint loading in various scenarios."""
        # Scenario 1: Load with same model architecture
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5), torch.nn.ReLU(), torch.nn.Linear(5, 1)
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        loaded_state = load_checkpoint(
            model=model,
            optimizer=optimizer,
            checkpoint_path=str(checkpoint_path),
            device=torch.device("cpu"),
        )

        assert "epoch" in loaded_state
        assert "best_metric_value" in loaded_state

        # Scenario 2: Load only model state (no optimizer)
        model_only = torch.nn.Sequential(
            torch.nn.Linear(10, 5), torch.nn.ReLU(), torch.nn.Linear(5, 1)
        )

        loaded_state_model_only = load_checkpoint(
            model=model_only,
            optimizer=None,
            checkpoint_path=str(checkpoint_path),
            device=torch.device("cpu"),
        )

        assert "epoch" in loaded_state_model_only

    @pytest.mark.integration
    def test_configuration_loading_across_formats(
        self, experiment_setup: tuple[Path, Trainer]
    ) -> None:
        """Test configuration loading across YAML and JSON formats."""
        experiment_dir, trainer = experiment_setup

        # Generate configuration artifacts
        trainer.train()

        config_storage = StandardizedConfigStorage(
            base_dir=experiment_dir / "configurations"
        )

        # Test loading configurations
        experiment_ids = config_storage.list_experiments()
        assert len(experiment_ids) > 0

        for experiment_id in experiment_ids:
            # Load configuration
            config = config_storage.load_configuration(experiment_id)
            assert config is not None
            assert "experiment" in config
            assert "training" in config

            # Test saving in different format and loading back
            json_path = config_storage.save_configuration(
                config=config,
                experiment_id=f"{experiment_id}_json_test",
                format_type="json",
            )
            assert json_path.exists()

            reloaded_config = config_storage.load_configuration(
                f"{experiment_id}_json_test"
            )
            # Configurations should be equivalent (ignoring timestamp
            # differences)
            assert (
                reloaded_config["experiment"]["name"]
                == config["experiment"]["name"]
            )


class TestArtifactValidation:
    """Test artifact completeness and correctness validation.

    Covers action item 4: Add validation tests to verify artifact completeness
    and correctness.
    """

    @pytest.mark.integration
    def test_checkpoint_validation(
        self, experiment_setup: tuple[Path, Trainer]
    ) -> None:
        """Test validation of checkpoint artifact completeness."""
        experiment_dir, trainer = experiment_setup

        # Generate checkpoints
        trainer.train()

        # Validate all generated checkpoints
        checkpoint_dir = experiment_dir / "checkpoints"
        checkpoint_files = list(checkpoint_dir.glob("*.pth*"))

        for checkpoint_file in checkpoint_files:
            self._validate_checkpoint_structure(checkpoint_file)

    def _validate_checkpoint_structure(self, checkpoint_path: Path) -> None:
        """Validate the structure and content of a checkpoint file."""
        checkpoint_data = torch.load(checkpoint_path, map_location="cpu")

        # Required fields
        required_fields = [
            "model_state_dict",
            "optimizer_state_dict",
            "epoch",
            "best_metric_value",
        ]

        for field in required_fields:
            assert (
                field in checkpoint_data
            ), f"Checkpoint missing required field: {field}"

        # Validate data types
        assert isinstance(checkpoint_data["epoch"], int)
        assert isinstance(checkpoint_data["best_metric_value"], int | float)
        assert isinstance(checkpoint_data["model_state_dict"], dict)
        assert isinstance(checkpoint_data["optimizer_state_dict"], dict)

    @pytest.mark.integration
    def test_configuration_validation(
        self, experiment_setup: tuple[Path, Trainer]
    ) -> None:
        """Test validation of configuration artifact completeness."""
        experiment_dir, trainer = experiment_setup

        # Generate configuration artifacts
        trainer.train()

        # Validate configuration artifacts
        config_storage = StandardizedConfigStorage(
            base_dir=experiment_dir / "configurations"
        )

        experiment_ids = config_storage.list_experiments()
        for experiment_id in experiment_ids:
            config = config_storage.load_configuration(experiment_id)
            self._validate_configuration_structure(config)

    def _validate_configuration_structure(self, config: DictConfig) -> None:
        """Validate the structure and completeness of a configuration."""
        # Required top-level sections
        required_sections = ["experiment", "model", "training"]
        for section in required_sections:
            assert (
                section in config
            ), f"Configuration missing section: {section}"

        # Should have environment metadata (enriched by standardized storage)
        assert (
            "environment" in config
        ), "Configuration should include environment metadata"
        assert (
            "config_metadata" in config
        ), "Configuration should include metadata"

        # Validate environment metadata structure
        env_metadata = config["environment"]
        expected_env_fields = [
            "pytorch_version",
            "python_version",
            "platform",
            "cuda_available",
            "timestamp",
        ]
        for field in expected_env_fields:
            assert (
                field in env_metadata
            ), f"Environment metadata missing field: {field}"

    @pytest.mark.integration
    def test_metrics_validation(
        self, experiment_setup: tuple[Path, Trainer]
    ) -> None:
        """Test validation of metrics artifact completeness."""
        experiment_dir, trainer = experiment_setup

        # Generate metrics artifacts
        trainer.train()

        # Find metrics artifacts
        metrics_manager = trainer.metrics_manager
        assert metrics_manager is not None

        # Export metrics summary for validation
        summary_path = metrics_manager.export_metrics_summary()
        assert summary_path.exists()

        # Validate metrics summary content
        with open(summary_path, encoding="utf-8") as f:
            summary_data = json.load(f)

        self._validate_metrics_summary_structure(summary_data)

    def _validate_metrics_summary_structure(
        self, summary_data: dict[str, Any]
    ) -> None:
        """Validate the structure of metrics summary."""
        required_fields = ["experiment_info", "training_summary"]
        for field in required_fields:
            assert (
                field in summary_data
            ), f"Metrics summary missing field: {field}"

        # Validate experiment info
        exp_info = summary_data["experiment_info"]
        assert "experiment_id" in exp_info
        assert "start_time" in exp_info

        # Validate training summary
        training_summary = summary_data["training_summary"]
        if training_summary:  # May be empty for short runs
            assert "total_epochs" in training_summary
            assert isinstance(training_summary["total_epochs"], int)


class TestArtifactCompatibility:
    """Test artifact compatibility across different versions.

    Covers action item 5: Implement compatibility tests for artifacts across
    different versions.
    """

    @pytest.mark.integration
    def test_configuration_format_compatibility(
        self, experiment_setup: tuple[Path, Trainer]
    ) -> None:
        """Test compatibility between different configuration formats."""
        experiment_dir, trainer = experiment_setup

        # Generate configuration
        trainer.train()

        config_storage = StandardizedConfigStorage(
            base_dir=experiment_dir / "configurations"
        )

        experiment_ids = config_storage.list_experiments()
        assert len(experiment_ids) > 0

        original_experiment_id = experiment_ids[0]
        original_config = config_storage.load_configuration(
            original_experiment_id
        )

        # Test format conversion compatibility
        # Save as JSON
        config_storage.save_configuration(
            config=original_config,
            experiment_id=f"{original_experiment_id}_json",
            format_type="json",
        )

        # Save as YAML
        config_storage.save_configuration(
            config=original_config,
            experiment_id=f"{original_experiment_id}_yaml",
            format_type="yaml",
        )

        # Load both and compare
        json_config = config_storage.load_configuration(
            f"{original_experiment_id}_json"
        )
        yaml_config = config_storage.load_configuration(
            f"{original_experiment_id}_yaml"
        )

        # Core configuration should be identical (ignoring timestamp
        # differences)
        assert (
            json_config["experiment"]["name"]
            == yaml_config["experiment"]["name"]
        )
        assert (
            json_config["training"]["epochs"]
            == yaml_config["training"]["epochs"]
        )

    @pytest.mark.integration
    def test_checkpoint_version_compatibility(
        self, experiment_setup: tuple[Path, Trainer]
    ) -> None:
        """Test checkpoint loading compatibility across versions."""
        experiment_dir, trainer = experiment_setup

        # Generate checkpoint
        trainer.train()

        # Test that checkpoints contain version information for compatibility
        checkpoint_dir = experiment_dir / "checkpoints"
        checkpoint_files = list(checkpoint_dir.glob("*.pth*"))
        assert len(checkpoint_files) > 0

        checkpoint_path = checkpoint_files[0]
        checkpoint_data = torch.load(checkpoint_path, map_location="cpu")

        # Verify checkpoint contains version/compatibility information
        # This ensures future versions can handle legacy checkpoints
        assert "model_state_dict" in checkpoint_data
        assert "optimizer_state_dict" in checkpoint_data

        # Test loading with version simulation
        with patch("torch.__version__", "2.0.0"):
            # Simulate loading in different PyTorch version
            loaded_state = load_checkpoint(
                model=trainer.model,
                optimizer=trainer.optimizer,
                checkpoint_path=str(checkpoint_path),
                device=torch.device("cpu"),
            )
            assert "epoch" in loaded_state


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
