"""Unit tests for ExperimentDataLoader."""

import json
import tempfile
from pathlib import Path

import pytest
import yaml
from omegaconf import DictConfig

from crackseg.reporting.data_loader import ExperimentDataLoader


class TestExperimentDataLoader:
    """Test ExperimentDataLoader functionality."""

    @pytest.fixture
    def data_loader(self) -> ExperimentDataLoader:
        """Provide ExperimentDataLoader instance."""
        return ExperimentDataLoader()

    @pytest.fixture
    def temp_experiment_dir(self):
        """Provide temporary experiment directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = Path(temp_dir) / "test_experiment"
            experiment_dir.mkdir()

            # Create config.yaml
            config = {
                "model": {"name": "swin_v2_b"},
                "data": {"dataset": "crack500"},
                "training": {"optimizer": "adamw", "epochs": 100},
            }
            with open(experiment_dir / "config.yaml", "w") as f:
                yaml.dump(config, f)

            # Create metrics directory
            metrics_dir = experiment_dir / "metrics"
            metrics_dir.mkdir()

            # Create complete_summary.json
            summary = {
                "best_epoch": 85,
                "best_iou": 0.78,
                "best_f1": 0.82,
                "best_precision": 0.85,
                "best_recall": 0.79,
                "final_loss": 0.12,
                "training_time": 3600.5,
            }
            with open(metrics_dir / "complete_summary.json", "w") as f:
                json.dump(summary, f)

            # Create metrics.jsonl
            epoch_metrics = [
                {"epoch": 1, "loss": 2.5, "iou": 0.3},
                {"epoch": 2, "loss": 2.1, "iou": 0.4},
            ]
            with open(metrics_dir / "metrics.jsonl", "w") as f:
                for metric in epoch_metrics:
                    f.write(json.dumps(metric) + "\n")

            # Create checkpoints directory
            checkpoints_dir = experiment_dir / "checkpoints"
            checkpoints_dir.mkdir()
            (checkpoints_dir / "model_best.pth").touch()
            (checkpoints_dir / "model_latest.pth").touch()

            # Create logs directory
            logs_dir = experiment_dir / "logs"
            logs_dir.mkdir()
            (logs_dir / "training.log").touch()

            yield experiment_dir

    def test_load_experiment_data_success(
        self, data_loader: ExperimentDataLoader, temp_experiment_dir: Path
    ) -> None:
        """Test successful experiment data loading."""
        experiment_data = data_loader.load_experiment_data(temp_experiment_dir)

        assert experiment_data.experiment_id == "test_experiment"
        assert experiment_data.experiment_dir == temp_experiment_dir
        assert isinstance(experiment_data.config, DictConfig)
        assert "complete_summary" in experiment_data.metrics
        assert len(experiment_data.artifacts) > 0
        assert "experiment_name" in experiment_data.metadata

    def test_load_experiment_data_missing_directory(
        self, data_loader: ExperimentDataLoader
    ) -> None:
        """Test loading from non-existent directory."""
        with pytest.raises(ValueError, match="does not exist"):
            data_loader.load_experiment_data(Path("/non/existent/path"))

    def test_load_experiment_data_not_directory(
        self, data_loader: ExperimentDataLoader
    ) -> None:
        """Test loading from path that is not a directory."""
        with tempfile.NamedTemporaryFile() as temp_file:
            with pytest.raises(ValueError, match="not a directory"):
                data_loader.load_experiment_data(Path(temp_file.name))

    def test_load_multiple_experiments(
        self, data_loader: ExperimentDataLoader, temp_experiment_dir: Path
    ) -> None:
        """Test loading multiple experiments."""
        # Create second experiment directory
        second_experiment_dir = (
            temp_experiment_dir.parent / "test_experiment_2"
        )
        second_experiment_dir.mkdir()

        # Create minimal config for second experiment
        config = {"model": {"name": "resnet50"}}
        with open(second_experiment_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)

        experiments_data = data_loader.load_multiple_experiments(
            [temp_experiment_dir, second_experiment_dir]
        )

        assert len(experiments_data) == 2
        assert experiments_data[0].experiment_id == "test_experiment"
        assert experiments_data[1].experiment_id == "test_experiment_2"

    def test_load_multiple_experiments_failure(
        self, data_loader: ExperimentDataLoader, temp_experiment_dir: Path
    ) -> None:
        """Test loading multiple experiments with one failure."""
        non_existent_dir = temp_experiment_dir.parent / "non_existent"

        with pytest.raises(ValueError, match="Failed to load experiment"):
            data_loader.load_multiple_experiments(
                [temp_experiment_dir, non_existent_dir]
            )

    def test_load_config_success(
        self, data_loader: ExperimentDataLoader, temp_experiment_dir: Path
    ) -> None:
        """Test successful config loading."""
        config = data_loader._load_config(temp_experiment_dir)

        assert isinstance(config, DictConfig)
        assert config.model.name == "swin_v2_b"
        assert config.data.dataset == "crack500"
        assert config.training.optimizer == "adamw"

    def test_load_config_missing_file(
        self, data_loader: ExperimentDataLoader, temp_experiment_dir: Path
    ) -> None:
        """Test config loading with missing config file."""
        # Remove config file
        (temp_experiment_dir / "config.yaml").unlink()

        config = data_loader._load_config(temp_experiment_dir)

        assert isinstance(config, DictConfig)
        assert len(config) == 0

    def test_load_config_invalid_yaml(
        self, data_loader: ExperimentDataLoader, temp_experiment_dir: Path
    ) -> None:
        """Test config loading with invalid YAML."""
        # Create invalid YAML
        with open(temp_experiment_dir / "config.yaml", "w") as f:
            f.write("invalid: yaml: content: [")

        with pytest.raises(ValueError, match="Invalid config file"):
            data_loader._load_config(temp_experiment_dir)

    def test_load_metrics_success(
        self, data_loader: ExperimentDataLoader, temp_experiment_dir: Path
    ) -> None:
        """Test successful metrics loading."""
        metrics = data_loader._load_metrics(temp_experiment_dir)

        assert "complete_summary" in metrics
        assert "epoch_metrics" in metrics
        assert metrics["complete_summary"]["best_iou"] == 0.78
        assert len(metrics["epoch_metrics"]) == 2

    def test_load_metrics_missing_directory(
        self, data_loader: ExperimentDataLoader, temp_experiment_dir: Path
    ) -> None:
        """Test metrics loading with missing metrics directory."""
        # Remove metrics directory
        import shutil

        shutil.rmtree(temp_experiment_dir / "metrics")

        metrics = data_loader._load_metrics(temp_experiment_dir)

        assert isinstance(metrics, dict)
        assert len(metrics) == 0

    def test_load_artifacts_success(
        self, data_loader: ExperimentDataLoader, temp_experiment_dir: Path
    ) -> None:
        """Test successful artifacts loading."""
        artifacts = data_loader._load_artifacts(temp_experiment_dir)

        assert "model_best.pth" in artifacts
        assert "model_latest.pth" in artifacts
        assert "log_training.log" in artifacts
        assert all(isinstance(path, Path) for path in artifacts.values())

    def test_load_artifacts_missing_directories(
        self, data_loader: ExperimentDataLoader, temp_experiment_dir: Path
    ) -> None:
        """Test artifacts loading with missing directories."""
        # Remove all artifact directories
        import shutil

        shutil.rmtree(temp_experiment_dir / "checkpoints")
        shutil.rmtree(temp_experiment_dir / "logs")

        artifacts = data_loader._load_artifacts(temp_experiment_dir)

        assert isinstance(artifacts, dict)
        assert len(artifacts) == 0

    def test_load_metadata_success(
        self, data_loader: ExperimentDataLoader, temp_experiment_dir: Path
    ) -> None:
        """Test successful metadata loading."""
        metadata = data_loader._load_metadata(temp_experiment_dir)

        assert "experiment_dir" in metadata
        assert "experiment_name" in metadata
        assert "config_summary" in metadata
        assert metadata["experiment_name"] == "test_experiment"
        assert metadata["config_summary"]["model"] == "swin_v2_b"

    def test_validate_experiment_structure_valid(
        self, data_loader: ExperimentDataLoader, temp_experiment_dir: Path
    ) -> None:
        """Test validation of valid experiment structure."""
        is_valid = data_loader.validate_experiment_structure(
            temp_experiment_dir
        )

        assert is_valid is True

    def test_validate_experiment_structure_missing_config(
        self, data_loader: ExperimentDataLoader, temp_experiment_dir: Path
    ) -> None:
        """Test validation with missing config file."""
        # Remove config file
        (temp_experiment_dir / "config.yaml").unlink()

        is_valid = data_loader.validate_experiment_structure(
            temp_experiment_dir
        )

        assert is_valid is False

    def test_validate_experiment_structure_no_optional_content(
        self, data_loader: ExperimentDataLoader
    ) -> None:
        """Test validation with no optional content directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = Path(temp_dir) / "minimal_experiment"
            experiment_dir.mkdir()

            # Create only config file
            config = {"model": {"name": "test"}}
            with open(experiment_dir / "config.yaml", "w") as f:
                yaml.dump(config, f)

            is_valid = data_loader.validate_experiment_structure(
                experiment_dir
            )

            assert is_valid is False

    def test_get_experiment_summary(
        self, data_loader: ExperimentDataLoader, temp_experiment_dir: Path
    ) -> None:
        """Test experiment summary generation."""
        experiment_data = data_loader.load_experiment_data(temp_experiment_dir)
        summary = data_loader.get_experiment_summary(experiment_data)

        assert summary["experiment_id"] == "test_experiment"
        assert summary["experiment_name"] == "test_experiment"
        assert "config_summary" in summary
        assert "metrics_summary" in summary
        assert "artifacts_count" in summary
        assert summary["metrics_summary"]["best_iou"] == 0.78
        assert summary["metrics_summary"]["best_f1"] == 0.82

    def test_get_experiment_summary_missing_metrics(
        self, data_loader: ExperimentDataLoader, temp_experiment_dir: Path
    ) -> None:
        """Test summary generation with missing metrics."""
        # Remove metrics directory
        import shutil

        shutil.rmtree(temp_experiment_dir / "metrics")

        experiment_data = data_loader.load_experiment_data(temp_experiment_dir)
        summary = data_loader.get_experiment_summary(experiment_data)

        assert summary["experiment_id"] == "test_experiment"
        assert summary["metrics_summary"] == {}
        # Artifacts are still present even without metrics
        assert summary["artifacts_count"] > 0
