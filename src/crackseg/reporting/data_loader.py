"""DataLoader implementation for loading experiment data from directories.

This module provides the DataLoader component that loads experiment
configuration, metrics, artifacts, and metadata from experiment directories.
"""

import logging
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from .config import ExperimentData
from .interfaces import DataLoader as DataLoaderInterface
from .utils.data_loading import (
    build_experiment_summary,
    load_artifacts_from_dir,
    load_config_from_dir,
    load_metadata_from_dir,
    load_metrics_from_dir,
)
from .utils.data_loading import (
    validate_experiment_structure as _validate_experiment_structure,
)


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
        return load_config_from_dir(experiment_dir)

    def _load_metrics(self, experiment_dir: Path) -> dict[str, Any]:
        """
        Load experiment metrics from metrics directory.

        Args:
            experiment_dir: Path to experiment directory

        Returns:
            Dictionary with metrics data
        """
        return load_metrics_from_dir(experiment_dir)

    def _load_artifacts(self, experiment_dir: Path) -> dict[str, Path]:
        """
        Load experiment artifacts from checkpoints and other directories.

        Args:
            experiment_dir: Path to experiment directory

        Returns:
            Dictionary mapping artifact names to file paths
        """
        return load_artifacts_from_dir(experiment_dir)

    def _load_metadata(self, experiment_dir: Path) -> dict[str, Any]:
        """
        Load experiment metadata from various sources.

        Args:
            experiment_dir: Path to experiment directory

        Returns:
            Dictionary with metadata information
        """
        return load_metadata_from_dir(experiment_dir)

    def validate_experiment_structure(self, experiment_dir: Path) -> bool:
        """
        Validate that an experiment directory has the expected structure.

        Args:
            experiment_dir: Path to experiment directory

        Returns:
            True if structure is valid, False otherwise
        """
        return _validate_experiment_structure(experiment_dir)

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
        return build_experiment_summary(experiment_data)
