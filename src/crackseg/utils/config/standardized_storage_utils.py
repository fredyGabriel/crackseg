"""Utilities for standardized configuration storage.

Extracted helpers to reduce module size and improve modularity.
Public API preserved via re-exports from standardized_storage.
"""

from __future__ import annotations

import hashlib
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf


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
        # Simplified detection to avoid runtime/version pitfalls
        metadata["cuda_version"] = "available"
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
    schema: Any | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Validate configuration against schema requirements."""
    from .standardized_storage import (
        ConfigurationSchema,  # local import to avoid cycle
    )

    if schema is None:
        schema = ConfigurationSchema()

    validation_result: dict[str, Any] = {
        "is_valid": True,
        "missing_required": [],
        "missing_recommended": [],
        "missing_environment": [],
        "completeness_score": 0.0,
    }

    for field_path in schema.required_fields:
        if not _has_nested_field(config, field_path):
            validation_result["missing_required"].append(field_path)
            validation_result["is_valid"] = False

    for field_path in schema.recommended_fields:
        if not _has_nested_field(config, field_path):
            validation_result["missing_recommended"].append(field_path)

    for field_path in schema.environment_fields:
        if not _has_nested_field(config, field_path):
            validation_result["missing_environment"].append(field_path)

    total_fields = len(schema.required_fields) + len(schema.recommended_fields)
    if total_fields > 0:
        missing_count = len(validation_result["missing_required"]) + len(
            validation_result["missing_recommended"]
        )
        validation_result["completeness_score"] = max(
            0.0, (total_fields - missing_count) / total_fields
        )

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
            if hasattr(current, "get"):
                value = current.get(key)
                if value is None:
                    return False
                current = value
            else:
                if not hasattr(current, key) or getattr(current, key) is None:
                    return False
                current = getattr(current, key)
        return True
    except (AttributeError, KeyError, TypeError):
        return False


def enrich_configuration_with_environment(
    config: DictConfig, include_environment: bool = True
) -> DictConfig:
    """Enrich configuration with automatic metadata and environment info."""
    enriched_config = OmegaConf.create(
        OmegaConf.to_container(config, resolve=True)
    )
    if not isinstance(enriched_config, DictConfig):
        raise ValueError("Expected DictConfig after conversion")

    config_metadata = {
        "created_at": datetime.now().isoformat(),
        "config_hash": _compute_config_hash(config),
        "format_version": "1.0",
    }
    enriched_config.config_metadata = config_metadata

    if include_environment:
        env_metadata = generate_environment_metadata()
        enriched_config.environment = env_metadata

    return enriched_config


def _compute_config_hash(config: DictConfig) -> str:
    """Compute a stable hash of the configuration for comparison."""
    config_str = OmegaConf.to_yaml(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def compare_configurations(
    config1: DictConfig,
    config2: DictConfig,
    ignore_fields: list[str] | None = None,
) -> dict[str, Any]:
    """Compare two configurations and identify differences."""
    if ignore_fields is None:
        ignore_fields = [
            "environment.timestamp",
            "config_metadata.created_at",
            "config_metadata.config_hash",
        ]

    clean_config1 = _remove_ignored_fields(config1, ignore_fields)
    clean_config2 = _remove_ignored_fields(config2, ignore_fields)

    flat1 = _flatten_config(clean_config1)
    flat2 = _flatten_config(clean_config2)

    all_keys = set(flat1.keys()) | set(flat2.keys())
    differences: dict[str, Any] = {}
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
    """Remove ignored fields from configuration for comparison."""
    config_copy = OmegaConf.create(
        OmegaConf.to_container(config, resolve=True)
    )
    if not isinstance(config_copy, DictConfig):
        raise ValueError("Expected DictConfig after conversion")

    for field_path in ignore_fields:
        keys = field_path.split(".")
        if len(keys) == 1:
            if keys[0] in config_copy:
                del config_copy[keys[0]]
        else:
            current = config_copy
            for key in keys[:-1]:
                if key in current and OmegaConf.is_dict(current[key]):
                    current = current[key]
                else:
                    break
            else:
                final_key = keys[-1]
                if final_key in current:
                    del current[final_key]
        continue

    return config_copy


def _flatten_config(config: DictConfig, prefix: str = "") -> dict[str, Any]:
    """Flatten nested configuration to dot-notation dictionary."""
    flat_dict: dict[str, Any] = {}
    for key, value in config.items():
        key_str = str(key)
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
    """Create a timestamped backup of configuration with metadata."""
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = (
        backup_dir / f"config_backup_{experiment_id}_{timestamp}.yaml"
    )
    enriched_config = enrich_configuration_with_environment(config)
    with open(backup_file, "w", encoding="utf-8") as f:
        OmegaConf.save(enriched_config, f)
    return backup_file


def migrate_legacy_configuration(
    legacy_config: dict[str, Any] | DictConfig,
    target_schema: Any | None = None,
) -> DictConfig:
    """Migrate a legacy configuration dictionary into standardized DictConfig."""
    if isinstance(legacy_config, dict):
        migrated_config = OmegaConf.create(legacy_config)
    else:
        config_dict = OmegaConf.to_container(legacy_config, resolve=True)
        migrated_config = OmegaConf.create(config_dict)

    # Ensure DictConfig type for downstream consumers
    if not isinstance(migrated_config, DictConfig):
        raise ValueError("Expected DictConfig after migration")
    enriched = enrich_configuration_with_environment(
        cast(DictConfig, migrated_config)
    )
    # Minimal migration metadata; callers can extend
    enriched.migration_metadata = {
        "migrated_at": datetime.now().isoformat(),
        "source_format": "legacy",
        "target_format": "standardized_v1.0",
    }
    return enriched
