"""
Artifact storage functionality.

This module contains the ArtifactStorage class responsible for saving
various types of artifacts with proper metadata tracking.
"""

import hashlib
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from .metadata import ArtifactMetadata

logger = logging.getLogger(__name__)


class ArtifactStorage:
    """Handles saving and loading of artifacts with metadata tracking."""

    def __init__(self, experiment_path: Path) -> None:
        """
        Initialize ArtifactStorage.

        Args:
            experiment_path: Path to the experiment directory
        """
        self.experiment_path = experiment_path

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file."""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate checksum for {file_path}: {e}")
            return ""

    def save_model(
        self,
        model: torch.nn.Module,
        filename: str,
        experiment_name: str,
        metadata: dict[str, Any] | None = None,
        **kwargs,
    ) -> tuple[str, ArtifactMetadata]:
        """
        Save trained model with metadata.

        Args:
            model: PyTorch model to save
            filename: Name of the model file
            experiment_name: Name of the experiment
            metadata: Additional metadata to store
            **kwargs: Additional arguments for torch.save

        Returns:
            Tuple of (file_path, metadata)
        """
        model_path = self.experiment_path / "models" / filename

        try:
            # Save model
            torch.save(model.state_dict(), model_path, **kwargs)

            # Create metadata
            meta = ArtifactMetadata(
                experiment_name=experiment_name,
                artifact_type="model",
                file_path=str(model_path),
                file_size=model_path.stat().st_size,
                checksum=self._calculate_checksum(model_path),
                description=f"Trained model: {filename}",
                tags=["model", "pytorch"],
                dependencies=[],
            )

            # Add custom metadata
            if metadata:
                for key, value in metadata.items():
                    setattr(meta, key, value)

            logger.info(f"Model saved: {model_path}")
            return str(model_path), meta

        except Exception as e:
            logger.error(f"Failed to save model {filename}: {e}")
            raise

    def save_metrics(
        self,
        metrics: dict[str, Any],
        filename: str,
        experiment_name: str,
        description: str = "",
    ) -> tuple[str, ArtifactMetadata]:
        """
        Save experiment metrics.

        Args:
            metrics: Dictionary of metrics to save
            filename: Name of the metrics file
            experiment_name: Name of the experiment
            description: Description of the metrics

        Returns:
            Tuple of (file_path, metadata)
        """
        metrics_path = self.experiment_path / "metrics" / filename

        try:
            # Add timestamp to metrics
            metrics_with_timestamp = {
                "timestamp": datetime.now().isoformat(),
                "experiment_name": experiment_name,
                **metrics,
            }

            # Save metrics
            with open(metrics_path, "w") as f:
                json.dump(metrics_with_timestamp, f, indent=2)

            # Create metadata
            meta = ArtifactMetadata(
                experiment_name=experiment_name,
                artifact_type="metrics",
                file_path=str(metrics_path),
                file_size=metrics_path.stat().st_size,
                checksum=self._calculate_checksum(metrics_path),
                description=description or f"Experiment metrics: {filename}",
                tags=["metrics", "json"],
                dependencies=[],
            )

            logger.info(f"Metrics saved: {metrics_path}")
            return str(metrics_path), meta

        except Exception as e:
            logger.error(f"Failed to save metrics {filename}: {e}")
            raise

    def save_config(
        self,
        config: dict[str, Any],
        filename: str,
        experiment_name: str,
        description: str = "",
    ) -> tuple[str, ArtifactMetadata]:
        """
        Save experiment configuration.

        Args:
            config: Configuration dictionary
            filename: Name of the config file
            experiment_name: Name of the experiment
            description: Description of the configuration

        Returns:
            Tuple of (file_path, metadata)
        """
        config_path = self.experiment_path / "configs" / filename

        try:
            # Add timestamp to config
            config_with_timestamp = {
                "timestamp": datetime.now().isoformat(),
                "experiment_name": experiment_name,
                **config,
            }

            # Save config
            with open(config_path, "w") as f:
                yaml.dump(config_with_timestamp, f, default_flow_style=False)

            # Create metadata
            meta = ArtifactMetadata(
                experiment_name=experiment_name,
                artifact_type="config",
                file_path=str(config_path),
                file_size=config_path.stat().st_size,
                checksum=self._calculate_checksum(config_path),
                description=description
                or f"Experiment configuration: {filename}",
                tags=["config", "yaml"],
                dependencies=[],
            )

            logger.info(f"Config saved: {config_path}")
            return str(config_path), meta

        except Exception as e:
            logger.error(f"Failed to save config {filename}: {e}")
            raise

    def save_visualization(
        self,
        file_path: str | Path,
        filename: str,
        experiment_name: str,
        description: str = "",
    ) -> tuple[str, ArtifactMetadata]:
        """
        Save visualization file.

        Args:
            file_path: Path to the visualization file
            filename: Name to save the file as
            experiment_name: Name of the experiment
            description: Description of the visualization

        Returns:
            Tuple of (file_path, metadata)
        """
        source_path = Path(file_path)
        target_path = self.experiment_path / "visualizations" / filename

        try:
            # Copy file to artifacts directory
            shutil.copy2(source_path, target_path)

            # Create metadata
            meta = ArtifactMetadata(
                experiment_name=experiment_name,
                artifact_type="visualization",
                file_path=str(target_path),
                file_size=target_path.stat().st_size,
                checksum=self._calculate_checksum(target_path),
                description=description or f"Visualization: {filename}",
                tags=["visualization", source_path.suffix[1:]],
                dependencies=[],
            )

            logger.info(f"Visualization saved: {target_path}")
            return str(target_path), meta

        except Exception as e:
            logger.error(f"Failed to save visualization {filename}: {e}")
            raise
