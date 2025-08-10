"""Standardized configuration storage and management utilities."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from crackseg.utils.logging import get_logger

from .standardized_storage_utils import (
    compare_configurations,
    enrich_configuration_with_environment,
    validate_configuration_completeness,
)

logger = get_logger(__name__)


@dataclass
class ConfigurationSchema:
    """Schema definition for standardized training configuration.

    This defines the required and optional fields for complete configuration
    storage and validation.
    """

    # Required core fields
    required_fields: list[str] = field(
        default_factory=lambda: [
            "experiment.name",
            "model._target_",
            "training.epochs",
            "training.optimizer._target_",
            "data.root_dir",
            "random_seed",
        ]
    )

    # Optional but recommended fields
    recommended_fields: list[str] = field(
        default_factory=lambda: [
            "training.learning_rate",
            "training.loss._target_",
            "data.batch_size",
            "model.encoder._target_",
            "model.decoder._target_",
            "evaluation.metrics",
        ]
    )

    # Environment fields (auto-generated)
    environment_fields: list[str] = field(
        default_factory=lambda: [
            "environment.pytorch_version",
            "environment.python_version",
            "environment.platform",
            "environment.cuda_available",
            "environment.cuda_version",
            "environment.timestamp",
        ]
    )


@dataclass
class StandardizedConfigStorage:
    """Standardized configuration storage manager."""

    base_dir: Path
    format_version: str = "1.0"
    include_environment: bool = True
    validate_on_save: bool = True

    def __post_init__(self) -> None:
        """Initialize storage manager."""
        self.base_dir = Path(self.base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_configuration(
        self,
        config: DictConfig,
        experiment_id: str,
        config_name: str = "config",
        format_type: str = "yaml",
    ) -> Path:
        """Save configuration with standardized format and validation.

        Args:
            config: Configuration to save
            experiment_id: Unique experiment identifier
            config_name: Name for the configuration file
            format_type: Format to save ('yaml' or 'json')

        Returns:
            Path to saved configuration file
        """
        validation_result: dict[str, Any] = {}

        # Validate configuration if requested
        if self.validate_on_save:
            validation_result = validate_configuration_completeness(config)
            if not validation_result["is_valid"]:
                missing_required = validation_result["missing_required"]
                logger.warning(
                    f"Configuration validation failed: {missing_required}"
                )

        # Enrich configuration with environment and metadata
        enriched_config = enrich_configuration_with_environment(
            config, self.include_environment
        )

        # Create experiment directory
        experiment_dir = self.base_dir / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)

        # Determine file extension and save method
        if format_type.lower() == "json":
            config_file = experiment_dir / f"{config_name}.json"
            self._save_as_json(enriched_config, config_file)
        else:  # Default to YAML
            config_file = experiment_dir / f"{config_name}.yaml"
            self._save_as_yaml(enriched_config, config_file)

        logger.info(f"Configuration saved to: {config_file}")

        # Save validation report
        if self.validate_on_save and validation_result:
            validation_file = experiment_dir / f"{config_name}_validation.json"
            with open(validation_file, "w", encoding="utf-8") as f:
                json.dump(validation_result, f, indent=2)

        return config_file

    def load_configuration(
        self, experiment_id: str, config_name: str = "config"
    ) -> DictConfig:
        """Load configuration from standardized storage.

        Args:
            experiment_id: Unique experiment identifier
            config_name: Name of the configuration file

        Returns:
            Loaded configuration
        """
        experiment_dir = self.base_dir / experiment_id

        # Try YAML first, then JSON
        yaml_file = experiment_dir / f"{config_name}.yaml"
        json_file = experiment_dir / f"{config_name}.json"

        if yaml_file.exists():
            loaded = OmegaConf.load(yaml_file)
            return cast(DictConfig, loaded)
        elif json_file.exists():
            with open(json_file, encoding="utf-8") as f:
                config_dict = json.load(f)
            return OmegaConf.create(config_dict)  # type: ignore[return-value]
        else:
            raise FileNotFoundError(
                f"Configuration not found for experiment {experiment_id}"
            )

    def compare_configurations(
        self,
        experiment_id1: str,
        experiment_id2: str,
        config_name: str = "config",
        ignore_fields: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compare configurations between two experiments.

        Args:
            experiment_id1: First experiment ID
            experiment_id2: Second experiment ID
            config_name: Configuration file name
            ignore_fields: Fields to ignore in comparison

        Returns:
            Comparison results
        """
        config1 = self.load_configuration(experiment_id1, config_name)
        config2 = self.load_configuration(experiment_id2, config_name)

        return compare_configurations(config1, config2, ignore_fields)

    def list_experiments(self) -> list[str]:
        """List all available experiment IDs."""
        if not self.base_dir.exists():
            return []

        return [
            d.name
            for d in self.base_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]

    def get_experiment_metadata(self, experiment_id: str) -> dict[str, Any]:
        """Get metadata for an experiment."""
        try:
            config = self.load_configuration(experiment_id)
            return {
                "experiment_id": experiment_id,
                "experiment_name": config.get("experiment", {}).get(
                    "name", "Unknown"
                ),
                "created_at": config.get("config_metadata", {}).get(
                    "created_at"
                ),
                "config_hash": config.get("config_metadata", {}).get(
                    "config_hash"
                ),
                "pytorch_version": config.get("environment", {}).get(
                    "pytorch_version"
                ),
                "platform": config.get("environment", {}).get("platform"),
            }
        except Exception as e:
            logger.warning(f"Failed to get metadata for {experiment_id}: {e}")
            return {"experiment_id": experiment_id, "error": str(e)}

    def _save_as_yaml(self, config: DictConfig, file_path: Path) -> None:
        """Save configuration as YAML."""
        with open(file_path, "w", encoding="utf-8") as f:
            OmegaConf.save(config, f)

    def _save_as_json(self, config: DictConfig, file_path: Path) -> None:
        """Save configuration as JSON."""
        config_dict = OmegaConf.to_container(config, resolve=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)


## Delegated to standardized_storage_utils


## Delegated to standardized_storage_utils


def _has_nested_field(config: DictConfig, field_path: str) -> bool:
    """Check if a nested field exists in configuration."""
    try:
        keys = field_path.split(".")
        current = config
        for key in keys:
            # For DictConfig objects, use get() method which returns None if key doesn't exist
            if hasattr(current, "get"):
                value = current.get(key)
                if value is None:
                    return False
                current = value
            else:
                # Fallback for regular objects
                if not hasattr(current, key) or getattr(current, key) is None:
                    return False
                current = getattr(current, key)
        return True
    except (AttributeError, KeyError, TypeError):
        return False


## Delegated to standardized_storage_utils


## Delegated to standardized_storage_utils


## Delegated to standardized_storage_utils


## Delegated to standardized_storage_utils


## Delegated to standardized_storage_utils


## Delegated to standardized_storage_utils


## Delegated to standardized_storage_utils
