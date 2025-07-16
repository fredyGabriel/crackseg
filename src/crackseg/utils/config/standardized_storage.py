"""Standardized configuration storage and management utilities."""

import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf

from crackseg.utils.logging import get_logger

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


def generate_environment_metadata() -> dict[str, Any]:
    """Generate environment metadata for configuration storage."""
    metadata: dict[str, Any] = {
        "pytorch_version": str(torch.__version__),
        "python_version": (
            f"{sys.version_info.major}.{sys.version_info.minor}."
            f"{sys.version_info.micro}"
        ),
        "platform": platform.platform(),
        "timestamp": datetime.now().isoformat(),
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        # Simplify CUDA version detection to avoid torch.version.cuda issues
        metadata["cuda_version"] = "available"  # Simplified approach
        metadata["cuda_device_count"] = str(torch.cuda.device_count())
        metadata["cuda_device_name"] = (
            torch.cuda.get_device_name(0)
            if torch.cuda.device_count() > 0
            else "none"
        )
    else:
        metadata["cuda_version"] = "not_available"
        metadata["cuda_device_count"] = "0"
        metadata["cuda_device_name"] = "not_available"

    return metadata


def validate_configuration_completeness(
    config: DictConfig,
    schema: ConfigurationSchema | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Validate configuration against schema requirements.

    Args:
        config: Configuration to validate
        schema: Schema defining required/recommended fields
        strict: Whether to raise exception on missing required fields

    Returns:
        Validation results with missing fields and status
    """
    if schema is None:
        schema = ConfigurationSchema()

    validation_result: dict[str, Any] = {
        "is_valid": True,
        "missing_required": [],
        "missing_recommended": [],
        "missing_environment": [],
        "completeness_score": 0.0,
    }

    # Check required fields
    for field_path in schema.required_fields:
        if not _has_nested_field(config, field_path):
            validation_result["missing_required"].append(field_path)
            validation_result["is_valid"] = False

    # Check recommended fields
    for field_path in schema.recommended_fields:
        if not _has_nested_field(config, field_path):
            validation_result["missing_recommended"].append(field_path)

    # Check environment fields
    for field_path in schema.environment_fields:
        if not _has_nested_field(config, field_path):
            validation_result["missing_environment"].append(field_path)

    # Calculate completeness score
    total_fields = len(schema.required_fields) + len(schema.recommended_fields)
    if total_fields > 0:
        missing_count = len(validation_result["missing_required"]) + len(
            validation_result["missing_recommended"]
        )
        validation_result["completeness_score"] = max(
            0.0, (total_fields - missing_count) / total_fields
        )

    # Strict validation
    if strict and not validation_result["is_valid"]:
        missing_fields = ", ".join(validation_result["missing_required"])
        raise ValueError(
            f"Configuration missing required fields: {missing_fields}"
        )

    return validation_result


def _has_nested_field(config: DictConfig, field_path: str) -> bool:
    """Check if a nested field exists in configuration."""
    try:
        keys = field_path.split(".")
        current = config
        for key in keys:
            if not hasattr(current, key) or current[key] is None:
                return False
            current = current[key]
        return True
    except (AttributeError, KeyError, TypeError):
        return False


def enrich_configuration_with_environment(
    config: DictConfig, include_environment: bool = True
) -> DictConfig:
    """Enrich configuration with automatic metadata and environment info.

    Args:
        config: Configuration to enrich
        include_environment: Whether to include environment metadata

    Returns:
        Enriched configuration
    """
    # Create a deep copy to avoid modifying the original
    enriched_config = OmegaConf.create(
        OmegaConf.to_container(config, resolve=True)
    )

    # Ensure we have a DictConfig
    if not isinstance(enriched_config, DictConfig):
        raise ValueError("Expected DictConfig after conversion")

    # Add configuration metadata
    config_metadata = {
        "created_at": datetime.now().isoformat(),
        "config_hash": _compute_config_hash(config),
        "format_version": "1.0",
    }
    enriched_config.config_metadata = config_metadata

    # Add environment metadata if requested
    if include_environment:
        env_metadata = generate_environment_metadata()
        enriched_config.environment = env_metadata

    return enriched_config


def _compute_config_hash(config: DictConfig) -> str:
    """Compute a hash of the configuration for comparison."""
    # Convert to sorted string representation for consistent hashing
    config_str = OmegaConf.to_yaml(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def compare_configurations(
    config1: DictConfig,
    config2: DictConfig,
    ignore_fields: list[str] | None = None,
) -> dict[str, Any]:
    """Compare two configurations and identify differences.

    Args:
        config1: First configuration
        config2: Second configuration
        ignore_fields: Fields to ignore in comparison (e.g., timestamps)

    Returns:
        Dictionary containing comparison results
    """
    if ignore_fields is None:
        ignore_fields = [
            "environment.timestamp",
            "config_metadata.created_at",
            "config_metadata.config_hash",
        ]

    # Create copies without ignored fields
    clean_config1 = _remove_ignored_fields(config1, ignore_fields)
    clean_config2 = _remove_ignored_fields(config2, ignore_fields)

    # Convert to flat dictionaries for easier comparison
    flat1 = _flatten_config(clean_config1)
    flat2 = _flatten_config(clean_config2)

    # Find differences
    all_keys = set(flat1.keys()) | set(flat2.keys())
    differences = {}

    for key in all_keys:
        val1 = flat1.get(key, "<MISSING>")
        val2 = flat2.get(key, "<MISSING>")

        if val1 != val2:
            differences[key] = {"config1": val1, "config2": val2}

    return {
        "are_identical": len(differences) == 0,
        "differences": differences,
        "total_differences": len(differences),
        "comparison_timestamp": datetime.now().isoformat(),
    }


def _remove_ignored_fields(
    config: DictConfig, ignore_fields: list[str]
) -> DictConfig:
    """Remove ignored fields from configuration for comparison.

    Args:
        config: Configuration to process
        ignore_fields: List of field paths to ignore

    Returns:
        Configuration with ignored fields removed
    """
    # Create a deep copy
    config_copy = OmegaConf.create(
        OmegaConf.to_container(config, resolve=True)
    )

    # Ensure we have a DictConfig
    if not isinstance(config_copy, DictConfig):
        raise ValueError("Expected DictConfig after conversion")

    # Remove ignored fields
    for field_path in ignore_fields:
        keys = field_path.split(".")
        if len(keys) == 1:
            # Top-level field
            if keys[0] in config_copy:
                del config_copy[keys[0]]
        else:
            # Nested field - navigate to parent and remove
            current = config_copy
            for key in keys[:-1]:
                if key in current and OmegaConf.is_dict(current[key]):
                    current = current[key]
                else:
                    break
            else:
                # Remove the final key if it exists
                final_key = keys[-1]
                if final_key in current:
                    del current[final_key]

        continue

    return config_copy


def _flatten_config(config: DictConfig, prefix: str = "") -> dict[str, Any]:
    """Flatten nested configuration to dot-notation dictionary."""
    flat_dict = {}

    for key, value in config.items():
        key_str = str(key)  # Convert key to string
        full_key = f"{prefix}.{key_str}" if prefix else key_str

        if OmegaConf.is_config(value):
            flat_dict.update(
                _flatten_config(cast(DictConfig, value), full_key)
            )
        else:
            flat_dict[full_key] = value

    return flat_dict


def create_configuration_backup(
    config: DictConfig, backup_dir: Path | str, experiment_id: str
) -> Path:
    """Create a backup of configuration with versioning.

    Args:
        config: Configuration to backup
        backup_dir: Directory for backups
        experiment_id: Experiment identifier

    Returns:
        Path to backup file
    """
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamped backup filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = (
        backup_dir / f"config_backup_{experiment_id}_{timestamp}.yaml"
    )

    # Save enriched configuration
    enriched_config = enrich_configuration_with_environment(config)
    with open(backup_file, "w", encoding="utf-8") as f:
        OmegaConf.save(enriched_config, f)

    logger.info(f"Configuration backup created: {backup_file}")
    return backup_file


def migrate_legacy_configuration(
    legacy_config: dict[str, Any] | DictConfig,
    target_schema: ConfigurationSchema | None = None,
) -> DictConfig:
    """Migrate legacy configuration to standardized format.

    Args:
        legacy_config: Legacy configuration to migrate
        target_schema: Target schema for migration

    Returns:
        Migrated configuration in standardized format
    """
    if target_schema is None:
        target_schema = ConfigurationSchema()

    # Convert to DictConfig if necessary
    if isinstance(legacy_config, dict):
        migrated_config = OmegaConf.create(legacy_config)
    else:
        # Create a new DictConfig from the existing one
        config_dict = OmegaConf.to_container(legacy_config, resolve=True)
        migrated_config = OmegaConf.create(config_dict)

    # Ensure we have a DictConfig (this check should always pass now)
    if not isinstance(migrated_config, DictConfig):
        raise ValueError(
            "Failed to create DictConfig from legacy configuration"
        )

    # Enrich with environment metadata - pass the DictConfig directly
    enriched = enrich_configuration_with_environment(migrated_config)

    # Add migration metadata
    enriched.migration_metadata = {
        "migrated_at": datetime.now().isoformat(),
        "source_format": "legacy",
        "target_format": "standardized_v1.0",
    }

    logger.info(
        "Successfully migrated legacy configuration to standardized format"
    )
    return enriched
