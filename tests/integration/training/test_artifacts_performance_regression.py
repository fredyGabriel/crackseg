"""Performance and regression tests for training artifacts.

This module covers action items 6-8 from subtask 9.4:
- CI/CD pipeline steps for artifact validation
- Performance tests for artifact loading and saving
- Regression tests comparing new and old artifact formats
"""

import json
import tempfile
import time
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

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
def performance_test_config() -> DictConfig:
    """Create configuration optimized for performance testing."""
    config_dict = {
        "experiment": {"name": "perf_test_experiment"},
        "model": {
            "_target_": "torch.nn.Linear",
            "in_features": 100,
            "out_features": 10,
        },
        "training": {
            "epochs": 5,
            "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.01},
            "scheduler": {
                "_target_": "torch.optim.lr_scheduler.StepLR",
                "step_size": 2,
                "gamma": 0.1,
            },
            "device": "cpu",
            "use_amp": False,
            "gradient_accumulation_steps": 1,
            "early_stopping": {"enabled": False},
            "save_freq": 1,  # Save every epoch for testing
        },
        "data": {"root_dir": "test_data", "batch_size": 32},
        "random_seed": 42,
    }
    return DictConfig(config_dict)


@pytest.fixture
def large_model_components() -> TrainingComponents:
    """Create training components with larger model for performance testing."""
    model = torch.nn.Sequential(
        torch.nn.Linear(100, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 10),
    )

    # Create larger datasets for performance testing
    train_loader = Mock()
    train_loader.__len__ = Mock(return_value=50)  # More batches
    train_loader.__iter__ = Mock(
        return_value=iter(
            [(torch.randn(32, 100), torch.randn(32, 10)) for _ in range(50)]
        )
    )

    val_loader = Mock()
    val_loader.__len__ = Mock(return_value=15)
    val_loader.__iter__ = Mock(
        return_value=iter(
            [(torch.randn(32, 100), torch.randn(32, 10)) for _ in range(15)]
        )
    )

    loss_fn = torch.nn.MSELoss()
    metrics_dict = {"mae": torch.nn.L1Loss(), "mse": torch.nn.MSELoss()}

    return TrainingComponents(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        metrics_dict=metrics_dict,
    )


class TestArtifactPerformance:
    """Test performance characteristics of artifact operations.

    Covers action item 7: Add performance tests for artifact loading
    and saving.
    """

    @pytest.mark.integration
    @pytest.mark.performance
    def test_checkpoint_save_load_performance(
        self,
        large_model_components: TrainingComponents,
        performance_test_config: DictConfig,
    ) -> None:
        """Test checkpoint save/load performance meets acceptable
        thresholds."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = Path(temp_dir) / "perf_test"
            experiment_dir.mkdir(parents=True)
            checkpoints_dir = experiment_dir / "checkpoints"
            checkpoints_dir.mkdir(parents=True)  # Ensure checkpoint dir exists

            # Set up trainer with simple mock to avoid nested mocking issues
            class SimpleExperimentManager:
                def __init__(
                    self,
                    experiment_dir: Path,
                    experiment_id: str,
                    checkpoint_dir: Path,
                ) -> None:
                    self.experiment_dir = experiment_dir
                    self.experiment_id = experiment_id
                    self._checkpoint_dir = checkpoint_dir

                def get_path(self, path_type: str) -> str:
                    return str(self._checkpoint_dir)

            mock_experiment_manager = SimpleExperimentManager(
                experiment_dir=experiment_dir,
                experiment_id="perf_test_001",
                checkpoint_dir=checkpoints_dir,
            )

            mock_logger = Mock()
            mock_logger.experiment_manager = mock_experiment_manager

            trainer = Trainer(
                components=large_model_components,
                cfg=performance_test_config,
                logger_instance=mock_logger,
            )

            # Run training to generate checkpoints
            start_time = time.time()
            trainer.train()
            training_time = time.time() - start_time

            # Performance benchmark: Training should complete within
            # reasonable time. For small model, should be under 30 seconds
            # on CPU
            assert (
                training_time < 30.0
            ), f"Training took {training_time:.2f}s, expected < 30s"

            # Test checkpoint loading performance
            checkpoint_dir = experiment_dir / "checkpoints"
            checkpoint_files = list(checkpoint_dir.glob("*.pth*"))
            assert len(checkpoint_files) > 0

            checkpoint_path = checkpoint_files[0]

            # Benchmark checkpoint loading
            load_times = []
            for _ in range(5):  # Average over 5 loads
                model = torch.nn.Sequential(
                    torch.nn.Linear(100, 256),
                    torch.nn.ReLU(),
                    torch.nn.Linear(256, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, 10),
                )
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

                start_time = time.time()
                load_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    checkpoint_path=str(checkpoint_path),
                    device=torch.device("cpu"),
                )
                load_time = time.time() - start_time
                load_times.append(load_time)

            avg_load_time = sum(load_times) / len(load_times)

            # Performance benchmark: Checkpoint loading should be under
            # 2 seconds
            assert avg_load_time < 2.0, (
                f"Average checkpoint load time {avg_load_time:.3f}s, "
                f"expected < 2s"
            )

    @pytest.mark.integration
    @pytest.mark.performance
    def test_configuration_storage_performance(
        self, performance_test_config: DictConfig
    ) -> None:
        """Test configuration storage performance meets acceptable
        thresholds."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_storage = StandardizedConfigStorage(
                base_dir=Path(temp_dir) / "config_perf_test"
            )

            # Benchmark configuration saving
            save_times = []
            for i in range(10):  # Test multiple saves
                start_time = time.time()
                config_storage.save_configuration(
                    config=performance_test_config,
                    experiment_id=f"perf_exp_{i:03d}",
                    format_type="yaml",
                )
                save_time = time.time() - start_time
                save_times.append(save_time)

            avg_save_time = sum(save_times) / len(save_times)

            # Performance benchmark: Config save should be under 0.5 seconds
            assert avg_save_time < 0.5, (
                f"Average config save time {avg_save_time:.3f}s, "
                f"expected < 0.5s"
            )

            # Benchmark configuration loading
            load_times = []
            for i in range(10):
                start_time = time.time()
                config_storage.load_configuration(f"perf_exp_{i:03d}")
                load_time = time.time() - start_time
                load_times.append(load_time)

            avg_load_time = sum(load_times) / len(load_times)

            # Performance benchmark: Config load should be under 0.2 seconds
            assert avg_load_time < 0.2, (
                f"Average config load time {avg_load_time:.3f}s, "
                f"expected < 0.2s"
            )

    @pytest.mark.integration
    @pytest.mark.performance
    def test_metrics_export_performance(
        self,
        large_model_components: TrainingComponents,
        performance_test_config: DictConfig,
    ) -> None:
        """Test metrics export performance meets acceptable thresholds."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = Path(temp_dir) / "metrics_perf_test"
            experiment_dir.mkdir(parents=True)
            metrics_dir = experiment_dir / "metrics"
            metrics_dir.mkdir(parents=True)

            # Use SimpleExperimentManager to avoid mocking issues
            class SimpleExperimentManager:
                def __init__(
                    self,
                    experiment_dir: Path,
                    experiment_id: str,
                ) -> None:
                    self.experiment_dir = experiment_dir
                    self.experiment_id = experiment_id

                def get_path(self, path_type: str) -> str:
                    if path_type == "checkpoints":
                        return str(self.experiment_dir / "checkpoints")
                    return str(self.experiment_dir / path_type)

            mock_experiment_manager = SimpleExperimentManager(
                experiment_dir=experiment_dir,
                experiment_id="metrics_perf_001",
            )

            mock_logger = Mock()
            mock_logger.experiment_manager = mock_experiment_manager

            trainer = Trainer(
                components=large_model_components,
                cfg=performance_test_config,
                logger_instance=mock_logger,
            )

            # Run training to generate metrics
            trainer.train()

            # Benchmark metrics export
            start_time = time.time()
            summary_path = trainer.metrics_manager.export_metrics_summary()
            export_time = time.time() - start_time

            # Performance benchmark: Metrics export should be under 1 second
            assert (
                export_time < 1.0
            ), f"Metrics export took {export_time:.3f}s, expected < 1s"

            # Verify exported file is not empty and well-formed
            assert summary_path.exists()
            assert summary_path.stat().st_size > 0

            with open(summary_path, encoding="utf-8") as f:
                summary_data = json.load(f)
            assert "experiment_info" in summary_data


class TestArtifactRegression:
    """Test regression scenarios for artifact formats and compatibility.

    Covers action item 8: Implement regression tests comparing new and old
    artifact formats.
    """

    @pytest.mark.integration
    @pytest.mark.regression
    def test_checkpoint_format_regression(
        self,
        large_model_components: TrainingComponents,
        performance_test_config: DictConfig,
    ) -> None:
        """Test that current checkpoint format is compatible with
        expected structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = Path(temp_dir) / "regression_test"
            experiment_dir.mkdir(parents=True)
            checkpoints_dir = experiment_dir / "checkpoints"
            checkpoints_dir.mkdir(parents=True)  # Ensure checkpoint dir exists

            # Set up trainer with simple mock to avoid nested mocking issues
            class SimpleExperimentManager:
                def __init__(
                    self,
                    experiment_dir: Path,
                    experiment_id: str,
                    checkpoint_dir: Path,
                ) -> None:
                    self.experiment_dir = experiment_dir
                    self.experiment_id = experiment_id
                    self._checkpoint_dir = checkpoint_dir

                def get_path(self, path_type: str) -> str:
                    return str(self._checkpoint_dir)

            mock_experiment_manager = SimpleExperimentManager(
                experiment_dir=experiment_dir,
                experiment_id="regression_001",
                checkpoint_dir=checkpoints_dir,
            )

            mock_logger = Mock()
            mock_logger.experiment_manager = mock_experiment_manager

            trainer = Trainer(
                components=large_model_components,
                cfg=performance_test_config,
                logger_instance=mock_logger,
            )

            # Generate checkpoint
            trainer.train()

            # Verify checkpoint format matches expected schema
            checkpoint_dir = experiment_dir / "checkpoints"
            checkpoint_files = list(checkpoint_dir.glob("*.pth*"))
            assert len(checkpoint_files) > 0

            checkpoint_path = checkpoint_files[0]
            checkpoint_data = torch.load(checkpoint_path, map_location="cpu")

            # Regression test: Verify expected checkpoint structure
            expected_checkpoint_schema = {
                "model_state_dict": dict,
                "optimizer_state_dict": dict,
                "epoch": int,
                "best_metric_value": (int, float),
            }

            for (
                field_name,
                expected_type,
            ) in expected_checkpoint_schema.items():
                assert (
                    field_name in checkpoint_data
                ), f"Checkpoint missing required field: {field_name}"
                assert isinstance(
                    checkpoint_data[field_name], expected_type
                ), (
                    f"Field {field_name} has type "
                    f"{type(checkpoint_data[field_name])}, "
                    f"expected {expected_type}"
                )

            # Regression test: Verify model state dict structure
            model_state = checkpoint_data["model_state_dict"]
            assert len(model_state) > 0, "Model state dict should not be empty"

            # Check that all model parameters are present
            actual_param_count = len(
                [k for k in model_state.keys() if "weight" in k or "bias" in k]
            )
            assert (
                actual_param_count > 0
            ), "Should have model parameters in checkpoint"

    @pytest.mark.integration
    @pytest.mark.regression
    def test_configuration_format_regression(
        self, performance_test_config: DictConfig
    ) -> None:
        """Test that configuration format maintains expected structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_storage = StandardizedConfigStorage(
                base_dir=Path(temp_dir) / "config_regression_test"
            )

            # Save configuration
            config_storage.save_configuration(
                config=performance_test_config,
                experiment_id="regression_config_001",
                format_type="yaml",
            )

            # Load and verify structure
            loaded_config = config_storage.load_configuration(
                "regression_config_001"
            )

            # Convert OmegaConf DictConfig to regular dict for type checking
            if hasattr(loaded_config, "__dict__") and hasattr(
                OmegaConf, "to_container"
            ):
                result = OmegaConf.to_container(loaded_config)
                assert isinstance(
                    result, dict
                ), "Configuration should be a dictionary"
                loaded_config_dict = cast(dict[str, Any], result)
            else:
                assert isinstance(
                    loaded_config, dict
                ), "Configuration should be a dictionary"
                loaded_config_dict = cast(dict[str, Any], loaded_config)

            # Regression test: Verify expected configuration schema
            expected_config_schema = {
                "experiment": dict,
                "model": dict,
                "training": dict,
                "data": dict,
                "environment": dict,  # Should be auto-added
                "config_metadata": dict,  # Should be auto-added
            }

            for section_name, expected_type in expected_config_schema.items():
                assert (
                    section_name in loaded_config_dict
                ), f"Configuration missing required section: {section_name}"
                assert isinstance(
                    loaded_config_dict[section_name], expected_type
                ), (
                    f"Section {section_name} has type "
                    f"{type(loaded_config_dict[section_name])}, "
                    f"expected {expected_type}"
                )

            # Regression test: Verify environment metadata structure
            env_metadata = loaded_config_dict["environment"]
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

            # Regression test: Verify config metadata structure
            config_metadata = loaded_config_dict["config_metadata"]
            expected_metadata_fields = [
                "created_at",
                "config_hash",
                "format_version",
            ]

            for field in expected_metadata_fields:
                assert (
                    field in config_metadata
                ), f"Config metadata missing field: {field}"

    @pytest.mark.integration
    @pytest.mark.regression
    def test_metrics_format_regression(
        self,
        large_model_components: TrainingComponents,
        performance_test_config: DictConfig,
    ) -> None:
        """Test that metrics format maintains expected structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = Path(temp_dir) / "metrics_regression_test"
            experiment_dir.mkdir(parents=True)

            # Use SimpleExperimentManager to avoid mocking issues
            class SimpleExperimentManager:
                def __init__(
                    self,
                    experiment_dir: Path,
                    experiment_id: str,
                ) -> None:
                    self.experiment_dir = experiment_dir
                    self.experiment_id = experiment_id

                def get_path(self, path_type: str) -> str:
                    if path_type == "checkpoints":
                        return str(self.experiment_dir / "checkpoints")
                    return str(self.experiment_dir / path_type)

            mock_experiment_manager = SimpleExperimentManager(
                experiment_dir=experiment_dir,
                experiment_id="metrics_regression_001",
            )

            mock_logger = Mock()
            mock_logger.experiment_manager = mock_experiment_manager

            trainer = Trainer(
                components=large_model_components,
                cfg=performance_test_config,
                logger_instance=mock_logger,
            )

            # Generate metrics
            trainer.train()

            # Export metrics summary
            summary_path = trainer.metrics_manager.export_metrics_summary()
            assert summary_path.exists()

            # Load and verify structure
            with open(summary_path, encoding="utf-8") as f:
                metrics_data = json.load(f)

            # Regression test: Verify expected metrics summary schema
            expected_metrics_schema = {
                "experiment_info": dict,
                "epoch_summaries": list,
                "best_metrics": dict,
                "available_metrics": dict,
            }

            for section_name, expected_type in expected_metrics_schema.items():
                assert (
                    section_name in metrics_data
                ), f"Metrics summary missing section: {section_name}"
                assert isinstance(metrics_data[section_name], expected_type), (
                    f"Section {section_name} has type "
                    f"{type(metrics_data[section_name])}, "
                    f"expected {expected_type}"
                )

            # Regression test: Verify experiment info structure
            exp_info = metrics_data["experiment_info"]
            expected_exp_info_fields = [
                "directory",
                "start_time",
                "total_epochs",
            ]

            for field in expected_exp_info_fields:
                assert (
                    field in exp_info
                ), f"Experiment info missing field: {field}"


class TestCIPipelineValidation:
    """Test CI/CD pipeline validation scenarios.

    Covers action item 6: Create CI/CD pipeline steps to run artifact
    validation.
    """

    @pytest.mark.integration
    @pytest.mark.ci
    def test_artifact_validation_pipeline(
        self,
        large_model_components: TrainingComponents,
        performance_test_config: DictConfig,
    ) -> None:
        """Test complete artifact validation pipeline for CI/CD."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = Path(temp_dir) / "ci_validation_test"
            experiment_dir.mkdir(parents=True)

            # Set up trainer with simple mock to avoid nested mocking issues
            class SimpleExperimentManager:
                def __init__(
                    self,
                    experiment_dir: Path,
                    experiment_id: str,
                    checkpoint_dir: Path,
                ) -> None:
                    self.experiment_dir = experiment_dir
                    self.experiment_id = experiment_id
                    self._checkpoint_dir = checkpoint_dir

                def get_path(self, path_type: str) -> str:
                    return str(self._checkpoint_dir)

            mock_experiment_manager = SimpleExperimentManager(
                experiment_dir=experiment_dir,
                experiment_id="ci_validation_001",
                checkpoint_dir=experiment_dir / "checkpoints",
            )

            mock_logger = Mock()
            mock_logger.experiment_manager = mock_experiment_manager

            trainer = Trainer(
                components=large_model_components,
                cfg=performance_test_config,
                logger_instance=mock_logger,
            )

            # Run complete training workflow
            trainer.train()

            # CI/CD Validation Step 1: Verify all expected artifacts exist
            validation_results = self._validate_artifact_completeness(
                experiment_dir
            )
            assert validation_results[
                "all_artifacts_present"
            ], f"Missing artifacts: {validation_results['missing_artifacts']}"

            # CI/CD Validation Step 2: Verify artifact formats are valid
            format_results = self._validate_artifact_formats(experiment_dir)
            assert format_results[
                "all_formats_valid"
            ], f"Invalid formats: {format_results['invalid_formats']}"

            # CI/CD Validation Step 3: Verify artifact loading works
            loading_results = self._validate_artifact_loading(
                experiment_dir, trainer
            )
            assert loading_results[
                "all_loading_successful"
            ], f"Loading failures: {loading_results['loading_failures']}"

    def _validate_artifact_completeness(
        self, experiment_dir: Path
    ) -> dict[str, Any]:
        """Validate that all expected artifacts are present."""
        expected_artifacts = {
            "checkpoints": ["*.pth*"],
            "configurations": [
                "**/training_config.*",
                "**/config_epoch_*.yaml",
            ],
            "metrics": ["**/complete_summary.json"],
        }

        missing_artifacts = []
        all_artifacts_present = True

        for artifact_type, patterns in expected_artifacts.items():
            for pattern in patterns:
                found_files = list(experiment_dir.glob(f"**/{pattern}"))
                if len(found_files) == 0:
                    missing_artifacts.append(f"{artifact_type}/{pattern}")
                    all_artifacts_present = False

        return {
            "all_artifacts_present": all_artifacts_present,
            "missing_artifacts": missing_artifacts,
        }

    def _validate_artifact_formats(
        self, experiment_dir: Path
    ) -> dict[str, Any]:
        """Validate that all artifacts have valid formats."""
        invalid_formats = []
        all_formats_valid = True

        # Validate checkpoint formats
        checkpoint_files = list(experiment_dir.glob("**/checkpoints/*.pth*"))
        for checkpoint_file in checkpoint_files:
            try:
                torch.load(checkpoint_file, map_location="cpu")
            except Exception as e:
                invalid_formats.append(
                    f"checkpoint/{checkpoint_file.name}: {e}"
                )
                all_formats_valid = False

        # Validate configuration formats
        config_files = list(experiment_dir.glob("**/configurations/**/*.yaml"))
        for config_file in config_files:
            try:
                with open(config_file, encoding="utf-8") as f:
                    yaml.safe_load(f)
            except Exception as e:
                invalid_formats.append(f"config/{config_file.name}: {e}")
                all_formats_valid = False

        # Validate metrics formats
        metrics_files = list(experiment_dir.glob("**/complete_summary.json"))
        for metrics_file in metrics_files:
            try:
                with open(metrics_file, encoding="utf-8") as f:
                    json.load(f)
            except Exception as e:
                invalid_formats.append(f"metrics/{metrics_file.name}: {e}")
                all_formats_valid = False

        return {
            "all_formats_valid": all_formats_valid,
            "invalid_formats": invalid_formats,
        }

    def _validate_artifact_loading(
        self, experiment_dir: Path, original_trainer: Trainer
    ) -> dict[str, Any]:
        """Validate that all artifacts can be loaded successfully."""
        loading_failures = []
        all_loading_successful = True

        # Test checkpoint loading
        checkpoint_files = list(experiment_dir.glob("**/checkpoints/*.pth*"))
        for checkpoint_file in checkpoint_files:
            try:
                load_checkpoint(
                    model=original_trainer.model,
                    optimizer=original_trainer.optimizer,
                    checkpoint_path=str(checkpoint_file),
                    device=torch.device("cpu"),
                )
            except Exception as e:
                loading_failures.append(
                    f"checkpoint/{checkpoint_file.name}: {e}"
                )
                all_loading_successful = False

        # Test configuration loading
        # Note: Configuration loading validation is covered by the format
        # validation above since we already verified all YAML files can be
        # loaded successfully

        return {
            "all_loading_successful": all_loading_successful,
            "loading_failures": loading_failures,
        }


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "integration"])
