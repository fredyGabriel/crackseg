"""Utility helpers for configuration integrity verification.

Extracted from ``config_verifier.py`` to reduce module size and improve
modularity while preserving the public API of ``ConfigIntegrityVerifier``.
"""

from __future__ import annotations

from typing import Any


def parse_basic_yaml(content: str) -> dict[str, Any]:
    """Parse a minimal YAML-like structure for basic validation.

    This is a simplified parser intended only for structure checks where
    installing a full YAML dependency is not desired.
    """
    config_data: dict[str, Any] = {}
    current_section: str | None = None

    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()

            if not value:  # Section header
                current_section = key
                config_data[current_section] = {}
            else:
                if current_section:
                    assert isinstance(config_data[current_section], dict)
                    config_data[current_section][key] = value
                else:
                    config_data[key] = value

    return config_data


def verify_section_structures(
    config_data: dict[str, Any], add_warning, add_metadata
) -> None:
    """Verify section-level structure, using provided callbacks.

    The callbacks mirror ``VerificationResult.add_warning`` and
    ``VerificationResult.add_metadata`` to avoid tight coupling.
    """
    try:
        # Model section
        if "model" in config_data:
            model_config = config_data["model"]
            if isinstance(model_config, dict):
                required_model_fields = ["encoder", "decoder"]
                missing_model_fields = [
                    field
                    for field in required_model_fields
                    if field not in model_config
                ]
                if missing_model_fields:
                    add_warning(
                        f"Missing recommended model fields: {missing_model_fields}"
                    )
                add_metadata("model_fields", list(model_config.keys()))

        # Training section
        if "training" in config_data:
            training_config = config_data["training"]
            if isinstance(training_config, dict):
                required_training_fields = [
                    "epochs",
                    "learning_rate",
                    "batch_size",
                ]
                missing_training_fields = [
                    field
                    for field in required_training_fields
                    if field not in training_config
                ]
                if missing_training_fields:
                    add_warning(
                        f"Missing recommended training fields: {missing_training_fields}"
                    )
                add_metadata("training_fields", list(training_config.keys()))

        # Data section
        if "data" in config_data:
            data_config = config_data["data"]
            if isinstance(data_config, dict):
                required_data_fields = ["image_size", "batch_size"]
                missing_data_fields = [
                    field
                    for field in required_data_fields
                    if field not in data_config
                ]
                if missing_data_fields:
                    add_warning(
                        f"Missing recommended data fields: {missing_data_fields}"
                    )
                add_metadata("data_fields", list(data_config.keys()))

        # Experiment section
        if "experiment" in config_data:
            experiment_config = config_data["experiment"]
            if isinstance(experiment_config, dict):
                required_experiment_fields = ["name", "output_dir"]
                missing_experiment_fields = [
                    field
                    for field in required_experiment_fields
                    if field not in experiment_config
                ]
                if missing_experiment_fields:
                    add_warning(
                        f"Missing recommended experiment fields: {missing_experiment_fields}"
                    )
                add_metadata(
                    "experiment_fields", list(experiment_config.keys())
                )

    except (
        Exception
    ) as exc:  # noqa: BLE001 - keep resilient during verification
        add_warning(f"Section structure verification failed: {exc}")


def calculate_dict_depth(data: dict[str, Any], current_depth: int = 0) -> int:
    """Calculate maximum depth of a nested dictionary."""
    if not isinstance(data, dict):
        return current_depth

    max_depth = current_depth
    for value in data.values():
        if isinstance(value, dict):
            depth = calculate_dict_depth(value, current_depth + 1)
            max_depth = max(max_depth, depth)
    return max_depth


def count_dict_keys(data: dict[str, Any]) -> int:
    """Count total keys in a nested dictionary."""
    if not isinstance(data, dict):
        return 0

    total_keys = len(data)
    for value in data.values():
        if isinstance(value, dict):
            total_keys += count_dict_keys(value)
    return total_keys


def find_nested_configs(config_data: dict[str, Any]) -> list[str]:
    """Find nested configuration references (paths to YAML/JSON)."""
    nested_configs: list[str] = []

    def _search_nested(data: Any, path: str = "") -> None:
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                if isinstance(value, str) and value.endswith(
                    (".yaml", ".yml", ".json")
                ):
                    nested_configs.append(current_path)
                elif isinstance(value, dict):
                    _search_nested(value, current_path)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                current_path = f"{path}[{i}]"
                _search_nested(item, current_path)

    _search_nested(config_data)
    return nested_configs


def find_environment_variables(config_data: dict[str, Any]) -> list[str]:
    """Find environment variable placeholders like ${VAR}."""
    env_vars: list[str] = []

    def _search_env_vars(data: Any) -> None:
        if (
            isinstance(data, str)
            and data.startswith("${")
            and data.endswith("}")
        ):
            env_vars.append(data[2:-1])  # Remove ${ and }
        elif isinstance(data, dict):
            for value in data.values():
                _search_env_vars(value)
        elif isinstance(data, list):
            for item in data:
                _search_env_vars(item)

    _search_env_vars(config_data)
    return env_vars


def check_circular_references(config_data: dict[str, Any]) -> list[str]:
    """Detect simple circular-style reference patterns by path reuse."""
    circular_refs: list[str] = []

    def _check_refs(data: Any, visited: set[str], path: str = "") -> None:
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                if current_path in visited:
                    circular_refs.append(current_path)
                else:
                    visited.add(current_path)
                    _check_refs(value, visited, current_path)
                    visited.remove(current_path)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                current_path = f"{path}[{i}]"
                _check_refs(item, visited, current_path)

    _check_refs(config_data, set())
    return circular_refs


def check_unused_references(config_data: dict[str, Any]) -> list[str]:
    """Placeholder for detecting unused references (non-invasive)."""
    # Intentionally conservative: return empty list to avoid false positives
    unused_refs: list[str] = []
    return unused_refs


def verify_path_consistency(config_data: dict[str, Any]) -> list[str]:
    """Detect relative path patterns that might be fragile in production."""
    path_issues: list[str] = []

    def _check_paths(data: Any, path: str = "") -> None:
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                if isinstance(value, str) and any(
                    keyword in key.lower()
                    for keyword in ["path", "dir", "file"]
                ):
                    if value.startswith("./") or value.startswith("../"):
                        path_issues.append(
                            f"{current_path}: relative path '{value}'"
                        )
                elif isinstance(value, dict):
                    _check_paths(value, current_path)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                current_path = f"{path}[{i}]"
                _check_paths(item, current_path)

    _check_paths(config_data)
    return path_issues
