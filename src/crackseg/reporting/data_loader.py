"""DataLoader implementation for loading experiment data from directories.

This module provides the DataLoader component that loads experiment
configuration, metrics, artifacts, and metadata from experiment directories.
"""

import json
import logging
from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf

from .config import ExperimentData
from .interfaces import DataLoader as DataLoaderInterface


class ExperimentDataLoader(DataLoaderInterface):
    """
    DataLoader implementation for loading experiment data from directories.

    This class loads experiment configuration, metrics, artifacts, and metadata
    from experiment directories following the CrackSeg experiment structure.
    """

    def __init__(self) -> None:
        """Initialize the ExperimentDataLoader."""
        self.logger = logging.getLogger(__name__)

    def load_experiment_data(self, experiment_dir: Path) -> ExperimentData:
        """
        Load experiment data from a single experiment directory.

        Args:
            experiment_dir: Path to experiment directory

        Returns:
            ExperimentData with loaded experiment information

        Raises:
            ValueError: If experiment directory is invalid or missing required
                files
        """
        self.logger.info(f"Loading experiment data from: {experiment_dir}")

        # Validate experiment directory
        if not experiment_dir.exists():
            raise ValueError(
                f"Experiment directory does not exist: {experiment_dir}"
            )

        if not experiment_dir.is_dir():
            raise ValueError(f"Path is not a directory: {experiment_dir}")

        # Load configuration
        config = self._load_config(experiment_dir)

        # Load metrics
        metrics = self._load_metrics(experiment_dir)

        # Load artifacts
        artifacts = self._load_artifacts(experiment_dir)

        # Load metadata
        metadata = self._load_metadata(experiment_dir)

        # Create experiment data
        experiment_data = ExperimentData(
            experiment_id=experiment_dir.name,
            experiment_dir=experiment_dir,
            config=config,
            metrics=metrics,
            artifacts=artifacts,
            metadata=metadata,
        )

        self.logger.info(
            f"Successfully loaded experiment: {experiment_data.experiment_id}"
        )
        return experiment_data

    def load_multiple_experiments(
        self, experiment_dirs: list[Path]
    ) -> list[ExperimentData]:
        """
        Load data from multiple experiment directories.

        Args:
            experiment_dirs: List of experiment directory paths

        Returns:
            List of ExperimentData objects

        Raises:
            ValueError: If any experiment directory is invalid
        """
        self.logger.info(f"Loading {len(experiment_dirs)} experiments")

        experiments_data = []

        for experiment_dir in experiment_dirs:
            try:
                experiment_data = self.load_experiment_data(experiment_dir)
                experiments_data.append(experiment_data)

            except Exception as e:
                self.logger.error(
                    f"Failed to load experiment {experiment_dir}: {e}"
                )
                raise ValueError(
                    f"Failed to load experiment {experiment_dir}: {e}"
                ) from e

        self.logger.info(
            f"Successfully loaded {len(experiments_data)} experiments"
        )
        return experiments_data

    def _load_config(self, experiment_dir: Path) -> DictConfig:
        """
        Load experiment configuration from config.yaml.

        Args:
            experiment_dir: Path to experiment directory

        Returns:
            DictConfig with experiment configuration

        Raises:
            ValueError: If config file is missing or invalid
        """
        config_path = experiment_dir / "config.yaml"

        if not config_path.exists():
            self.logger.warning(f"Config file not found: {config_path}")
            # Return empty config
            return OmegaConf.create({})

        try:
            with open(config_path, encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)

            if config_dict is None:
                config_dict = {}

            return OmegaConf.create(config_dict)

        except Exception as e:
            self.logger.error(f"Failed to load config from {config_path}: {e}")
            raise ValueError(f"Invalid config file {config_path}: {e}") from e

    def _load_metrics(self, experiment_dir: Path) -> dict[str, Any]:
        """
        Load experiment metrics from metrics directory.

        Args:
            experiment_dir: Path to experiment directory

        Returns:
            Dictionary with metrics data
        """
        metrics_data = {}
        metrics_dir = experiment_dir / "metrics"

        if not metrics_dir.exists():
            self.logger.warning(f"Metrics directory not found: {metrics_dir}")
            return metrics_data

        # Load complete summary
        summary_path = metrics_dir / "complete_summary.json"
        if summary_path.exists():
            try:
                with open(summary_path, encoding="utf-8") as f:
                    metrics_data["complete_summary"] = json.load(f)
            except Exception as e:
                self.logger.error(
                    f"Failed to load summary from {summary_path}: {e}"
                )

        # Load per-epoch metrics
        metrics_file = metrics_dir / "metrics.jsonl"
        if metrics_file.exists():
            try:
                epoch_metrics = []
                with open(metrics_file, encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            epoch_metrics.append(json.loads(line))
                metrics_data["epoch_metrics"] = epoch_metrics
            except Exception as e:
                self.logger.error(
                    f"Failed to load epoch metrics from {metrics_file}: {e}"
                )

        # Load validation metrics
        val_metrics_file = metrics_dir / "validation_metrics.json"
        if val_metrics_file.exists():
            try:
                with open(val_metrics_file, encoding="utf-8") as f:
                    metrics_data["validation_metrics"] = json.load(f)
            except Exception as e:
                self.logger.error(
                    f"Failed to load validation metrics from "
                    f"{val_metrics_file}: {e}"
                )

        # Load test metrics
        test_metrics_file = metrics_dir / "test_metrics.json"
        if test_metrics_file.exists():
            try:
                with open(test_metrics_file, encoding="utf-8") as f:
                    metrics_data["test_metrics"] = json.load(f)
            except Exception as e:
                self.logger.error(
                    f"Failed to load test metrics from "
                    f"{test_metrics_file}: {e}"
                )

        return metrics_data

    def _load_artifacts(self, experiment_dir: Path) -> dict[str, Path]:
        """
        Load experiment artifacts from checkpoints and other directories.

        Args:
            experiment_dir: Path to experiment directory

        Returns:
            Dictionary mapping artifact names to file paths
        """
        artifacts = {}

        # Load model checkpoints
        checkpoints_dir = experiment_dir / "checkpoints"
        if checkpoints_dir.exists():
            checkpoint_files = [
                "model_best.pth",
                "model_latest.pth",
                "optimizer_best.pth",
                "optimizer_latest.pth",
                "scheduler_best.pth",
                "scheduler_latest.pth",
            ]

            for checkpoint_file in checkpoint_files:
                checkpoint_path = checkpoints_dir / checkpoint_file
                if checkpoint_path.exists():
                    artifacts[checkpoint_file] = checkpoint_path

        # Load training logs
        logs_dir = experiment_dir / "logs"
        if logs_dir.exists():
            log_files = ["training.log", "validation.log", "test.log"]
            for log_file in log_files:
                log_path = logs_dir / log_file
                if log_path.exists():
                    artifacts[f"log_{log_file}"] = log_path

        # Load visualizations
        viz_dir = experiment_dir / "visualizations"
        if viz_dir.exists():
            viz_files = list(viz_dir.glob("*.png")) + list(
                viz_dir.glob("*.jpg")
            )
            for viz_file in viz_files:
                artifacts[f"viz_{viz_file.name}"] = viz_file

        # Load predictions
        predictions_dir = experiment_dir / "predictions"
        if predictions_dir.exists():
            pred_files = list(predictions_dir.glob("*.png")) + list(
                predictions_dir.glob("*.jpg")
            )
            for pred_file in pred_files:
                artifacts[f"pred_{pred_file.name}"] = pred_file

        return artifacts

    def _load_metadata(self, experiment_dir: Path) -> dict[str, Any]:
        """
        Load experiment metadata from various sources.

        Args:
            experiment_dir: Path to experiment directory

        Returns:
            Dictionary with metadata information
        """
        metadata: dict[str, Any] = {
            "experiment_dir": str(experiment_dir),
            "experiment_name": experiment_dir.name,
        }

        # Load experiment info
        info_file = experiment_dir / "experiment_info.json"
        if info_file.exists():
            try:
                with open(info_file, encoding="utf-8") as f:
                    metadata.update(json.load(f))
            except Exception as e:
                self.logger.error(
                    f"Failed to load experiment info from {info_file}: {e}"
                )

        # Load git information
        git_file = experiment_dir / "git_info.json"
        if git_file.exists():
            try:
                with open(git_file, encoding="utf-8") as f:
                    metadata["git_info"] = json.load(f)
            except Exception as e:
                self.logger.error(
                    f"Failed to load git info from {git_file}: {e}"
                )

        # Load system information
        sys_file = experiment_dir / "system_info.json"
        if sys_file.exists():
            try:
                with open(sys_file, encoding="utf-8") as f:
                    metadata["system_info"] = json.load(f)
            except Exception as e:
                self.logger.error(
                    f"Failed to load system info from {sys_file}: {e}"
                )

        # Extract metadata from config
        config_path = experiment_dir / "config.yaml"
        if config_path.exists():
            try:
                with open(config_path, encoding="utf-8") as f:
                    config_dict = yaml.safe_load(f)
                    if config_dict:
                        metadata["config_summary"] = {
                            "model": config_dict.get("model", {}).get(
                                "name", "unknown"
                            ),
                            "dataset": config_dict.get("data", {}).get(
                                "dataset", "unknown"
                            ),
                            "optimizer": config_dict.get("training", {}).get(
                                "optimizer", "unknown"
                            ),
                            "epochs": config_dict.get("training", {}).get(
                                "epochs", 0
                            ),
                        }
            except Exception as e:
                self.logger.error(
                    f"Failed to extract config metadata from "
                    f"{config_path}: {e}"
                )

        return metadata

    def validate_experiment_structure(self, experiment_dir: Path) -> bool:
        """
        Validate that an experiment directory has the expected structure.

        Args:
            experiment_dir: Path to experiment directory

        Returns:
            True if structure is valid, False otherwise
        """
        required_files = [
            "config.yaml",
        ]

        optional_dirs = [
            "metrics",
            "checkpoints",
            "logs",
            "visualizations",
            "predictions",
        ]

        # Check required files
        for required_file in required_files:
            if not (experiment_dir / required_file).exists():
                self.logger.warning(f"Missing required file: {required_file}")
                return False

        # Check optional directories (at least one should exist)
        has_optional_content = False
        for optional_dir in optional_dirs:
            if (experiment_dir / optional_dir).exists():
                has_optional_content = True
                break

        if not has_optional_content:
            self.logger.warning("No optional content directories found")
            return False

        return True

    def get_experiment_summary(
        self, experiment_data: ExperimentData
    ) -> dict[str, Any]:
        """
        Generate a summary of experiment data for reporting.

        Args:
            experiment_data: Loaded experiment data

        Returns:
            Dictionary with experiment summary
        """
        summary = {
            "experiment_id": experiment_data.experiment_id,
            "experiment_name": experiment_data.metadata.get(
                "experiment_name", "unknown"
            ),
            "config_summary": experiment_data.metadata.get(
                "config_summary", {}
            ),
            "metrics_summary": {},
            "artifacts_count": len(experiment_data.artifacts),
        }

        # Extract key metrics
        if "complete_summary" in experiment_data.metrics:
            complete_summary = experiment_data.metrics["complete_summary"]
            summary["metrics_summary"] = {
                "best_epoch": complete_summary.get("best_epoch", 0),
                "best_iou": complete_summary.get("best_iou", 0.0),
                "best_f1": complete_summary.get("best_f1", 0.0),
                "best_precision": complete_summary.get("best_precision", 0.0),
                "best_recall": complete_summary.get("best_recall", 0.0),
                "final_loss": complete_summary.get("final_loss", 0.0),
                "training_time": complete_summary.get("training_time", 0.0),
            }

        return summary
