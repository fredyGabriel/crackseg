"""
Experiment manager for organizing experiment outputs.

This module provides a centralized management of experiment directories,
ensuring a consistent structure across all runs.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf


class ExperimentManager:
    """
    Manages experiment directories and provides a consistent structure.

    This class handles:
    - Directory creation with standardized structure
    - Path resolution for different artifacts (checkpoints, metrics, etc.)
    - Registry of experiments for easier navigation
    """

    def __init__(  # noqa: PLR0913
        self,
        base_dir: str | Path = "outputs",
        experiment_name: str = "default",
        config: DictConfig | None = None,
        create_dirs: bool = True,
        timestamp_format: str = "%Y%m%d-%H%M%S",
        timestamp: str | None = None,
    ):
        """
        Initialize the experiment manager.

        Args:
            base_dir: Base directory for all experiments
            experiment_name: Name of the experiment
            config: Configuration for the experiment
            create_dirs: Whether to create directories immediately
            timestamp_format: Format for timestamp in directory name
            timestamp: Optional pre-defined timestamp to use (instead of
                       generating a new one)
        """
        self.base_dir = Path(base_dir)
        self.experiment_name = experiment_name
        self.config = config

        # Generate experiment ID with timestamp
        if timestamp:
            self.timestamp = timestamp
        else:
            self.timestamp = datetime.now().strftime(timestamp_format)

        self.experiment_id = f"{self.timestamp}-{experiment_name}"

        # Define directory structure - always use 'experiments' subfolder
        self.experiments_dir = self.base_dir / "experiments"
        self.experiment_dir = self.experiments_dir / self.experiment_id

        # Define subdirectories within experiment
        self.config_dir = self.experiment_dir
        self.checkpoints_dir = self.experiment_dir / "checkpoints"
        self.metrics_dir = self.experiment_dir / "metrics"
        self.logs_dir = self.experiment_dir / "logs"
        self.results_dir = self.experiment_dir / "results"
        self.visualizations_dir = self.results_dir / "visualizations"
        self.predictions_dir = self.results_dir / "predictions"

        # Define shared directories
        self.shared_dir = self.base_dir / "shared"
        self.registry_file = self.base_dir / "experiment_registry.json"

        # Create directories if requested
        if create_dirs:
            self._create_directory_structure()
            self._register_experiment()

        # Save paths as dict for easy access
        self.paths = {
            "base": self.base_dir,
            "experiments": self.experiments_dir,
            "experiment": self.experiment_dir,
            "config": self.config_dir,
            "checkpoints": self.checkpoints_dir,
            "metrics": self.metrics_dir,
            "logs": self.logs_dir,
            "results": self.results_dir,
            "visualizations": self.visualizations_dir,
            "predictions": self.predictions_dir,
            "shared": self.shared_dir,
        }

    def _create_directory_structure(self) -> None:
        """Create the directory structure for the experiment."""
        # Create experiment directory and all subdirectories
        for dir_path in [
            self.experiments_dir,
            self.experiment_dir,
            self.checkpoints_dir,
            self.metrics_dir,
            self.logs_dir,
            self.results_dir,
            self.visualizations_dir,
            self.predictions_dir / "validation",
            self.predictions_dir / "test",
            self.shared_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Create experiment info file
        self._save_experiment_info()

    def _save_experiment_info(self) -> None:
        """Save experiment information to a JSON file."""
        info = {
            "id": self.experiment_id,
            "name": self.experiment_name,
            "timestamp": self.timestamp,
            "created_at": datetime.now().isoformat(),
            "status": "created",
        }

        # Add configuration if available
        if self.config is not None:
            if isinstance(self.config, DictConfig):
                config_dict = OmegaConf.to_container(self.config, resolve=True)
                info["config"] = str(config_dict)
            elif self.config is not None:
                info["config"] = str(self.config)

        # Save as JSON
        with open(
            self.experiment_dir / "experiment_info.json", "w", encoding="utf-8"
        ) as f:
            json.dump(info, f, indent=2)

    def _register_experiment(self) -> None:
        """Register this experiment in the central registry."""
        registry = []

        # Load existing registry if it exists
        if self.registry_file.exists():
            try:
                with open(self.registry_file, encoding="utf-8") as f:
                    registry = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                print(f"Warning: Could not load experiment registry: {e}")
                registry = []
            except Exception as e:
                print(
                    "Unexpected error loading experiment registry: "
                    f"{type(e).__name__} - {e}"
                )
                registry = []

        # Add this experiment
        registry.append(
            {
                "id": self.experiment_id,
                "name": self.experiment_name,
                "timestamp": self.timestamp,
                "path": str(self.experiment_dir),
                "created_at": datetime.now().isoformat(),
            }
        )

        # Save registry
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)

    def get_path(self, key: str) -> Path:
        """
        Get a specific path from the experiment directory structure.

        Args:
            key: Path key (e.g., 'checkpoints', 'metrics')

        Returns:
            Path object for the requested directory

        Raises:
            KeyError: If the key is not a valid path
        """
        if key not in self.paths:
            raise KeyError(
                f"Unknown path key: {key}. Valid keys: \
{list(self.paths.keys())}"
            )

        return self.paths[key]

    def save_config(self, config: dict | DictConfig) -> Path:
        """
        Save configuration to a JSON file.

        Args:
            config: Configuration to save

        Returns:
            Path to the saved configuration file
        """
        # Convert OmegaConf to dict if needed
        if isinstance(config, DictConfig):
            config_dict = OmegaConf.to_container(config, resolve=True)
        else:
            config_dict = config

        # Save to JSON file
        config_path = self.config_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)

        return config_path

    def save_metrics_summary(self, metrics: dict[str, float]) -> Path:
        """
        Save metrics summary to a JSON file.

        Args:
            metrics: Dictionary of metrics

        Returns:
            Path to the saved metrics summary file
        """
        # Ensure metrics directory exists
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # Add timestamp
        metrics_dict = {
            **metrics,
            "timestamp": datetime.now().isoformat(),
            "experiment_id": self.experiment_id,
        }

        # Save to JSON file
        metrics_path = self.metrics_dir / "metrics_summary.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, indent=2)

        return metrics_path

    def update_status(
        self, status: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """
        Update experiment status in experiment_info.json.

        Args:
            status: New status (e.g., 'running', 'completed', 'failed')
            metadata: Additional metadata to save
        """
        info_path = self.experiment_dir / "experiment_info.json"
        info = {}

        # Load existing info if it exists
        if info_path.exists():
            try:
                with open(info_path, encoding="utf-8") as f:
                    info = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                print(f"Warning: Could not load experiment info: {e}")
                info = {
                    "id": self.experiment_id,
                    "name": self.experiment_name,
                    "timestamp": self.timestamp,
                    "created_at": datetime.now().isoformat(),
                }

        # Update status and metadata
        info["status"] = status
        info["updated_at"] = datetime.now().isoformat()

        if metadata:
            meta = info.get("metadata")
            if not isinstance(meta, dict):
                meta = {}
            meta.update(metadata)
            info["metadata"] = json.dumps(meta)

        # Save updated info
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)

    def get_available_experiments(self) -> list[dict[str, Any]]:
        """
        Get list of available experiments from registry.

        Returns:
            List of experiment information dictionaries
        """
        if not self.registry_file.exists():
            return []

        try:
            with open(self.registry_file, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and all(
                    isinstance(item, dict) for item in data
                ):
                    return data  # type: ignore
                else:
                    return []
        except (OSError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load experiment registry: {e}")
            return []
        except Exception as e:
            print(
                "Unexpected error reading experiment registry: "
                f"{type(e).__name__} - {e}"
            )
            return []

    @classmethod
    def get_experiment(
        cls, experiment_id: str, base_dir: str | Path = "outputs"
    ) -> Optional["ExperimentManager"]:
        """
        Get an experiment by ID.

        Args:
            experiment_id: ID of the experiment
            base_dir: Base directory for all experiments

        Returns:
            ExperimentManager instance for the experiment, or None if not found
        """
        # Find the experiment in the registry
        registry_file = Path(base_dir) / "experiment_registry.json"

        if not registry_file.exists():
            return None

        try:
            with open(registry_file, encoding="utf-8") as f:
                registry = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            # Log and return None if registry cannot be read
            # Assuming a logger might be available at cls.logger or a global
            # one
            print(
                "Warning: Could not load experiment registry to find "
                f"experiment '{experiment_id}': {e}"
            )
            return None
        except Exception as e:
            # Assuming a logger might be available at cls.logger or a global
            # one
            print(
                "Unexpected error reading experiment registry for "
                f"'{experiment_id}': {e}"
            )
            return None

        # Find the experiment
        for exp in registry:
            if exp["id"] == experiment_id:
                # Initialize ExperimentManager with existing path
                return cls(
                    base_dir=base_dir,
                    experiment_name=exp["name"],
                    create_dirs=False,
                    timestamp=exp.get("timestamp"),
                )

        return None
