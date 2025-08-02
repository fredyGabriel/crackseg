"""
Configuration integrity verification.

This module provides specialized integrity verification for configuration
files, integrating with the existing configuration validation system.
"""

import json
import logging
from pathlib import Path
from typing import Any

from .core import IntegrityVerifier, VerificationLevel, VerificationResult

logger = logging.getLogger(__name__)


class ConfigIntegrityVerifier(IntegrityVerifier):
    """Specialized verifier for configuration file integrity."""

    def __init__(
        self,
        verification_level: VerificationLevel = VerificationLevel.STANDARD,
        required_sections: list[str] | None = None,
    ):
        """
        Initialize configuration integrity verifier.

        Args:
            verification_level: Level of verification thoroughness
            required_sections: List of required configuration sections
        """
        super().__init__(verification_level)
        self.required_sections = required_sections or [
            "model",
            "training",
            "data",
            "experiment",
        ]

    def verify(self, artifact_path: Path) -> VerificationResult:
        """
        Verify the integrity of a configuration file.

        Args:
            artifact_path: Path to the configuration file

        Returns:
            VerificationResult with verification details
        """
        result = VerificationResult(
            is_valid=True,
            artifact_path=artifact_path,
            verification_level=self.verification_level,
        )

        # Basic structure verification
        if not self._verify_basic_structure(artifact_path, result):
            return result

        # Calculate checksum
        checksum = self._verify_checksum(artifact_path)
        result.checksum = checksum

        # Verify file extension
        if not self._verify_config_extension(artifact_path, result):
            return result

        if self.verification_level in [
            VerificationLevel.STANDARD,
            VerificationLevel.THOROUGH,
            VerificationLevel.PARANOID,
        ]:
            # Content validation
            if not self._verify_config_content(artifact_path, result):
                return result

        if self.verification_level in [
            VerificationLevel.THOROUGH,
            VerificationLevel.PARANOID,
        ]:
            # Deep content analysis
            self._verify_config_structure(artifact_path, result)

        if self.verification_level == VerificationLevel.PARANOID:
            # Cross-reference validation
            self._verify_config_consistency(artifact_path, result)

        return result

    def _verify_config_extension(
        self, config_path: Path, result: VerificationResult
    ) -> bool:
        """Verify that the file has a supported configuration extension."""
        file_extension = config_path.suffix.lower()
        supported_extensions = [".yaml", ".yml", ".json"]

        if file_extension not in supported_extensions:
            result.add_error(
                f"Unsupported configuration extension: {file_extension}"
            )
            return False

        result.add_metadata("config_format", file_extension)
        return True

    def _verify_config_content(
        self, config_path: Path, result: VerificationResult
    ) -> bool:
        """Verify configuration content and structure."""
        try:
            file_extension = config_path.suffix.lower()

            if file_extension == ".json":
                with open(config_path, encoding="utf-8") as f:
                    config_data = json.load(f)
            else:
                # For YAML files, we'll just check if they can be read
                # Full YAML parsing would require PyYAML dependency
                with open(config_path, encoding="utf-8") as f:
                    content = f.read()

                if not content.strip():
                    result.add_error("Configuration file is empty")
                    return False

                # Basic YAML structure check
                config_data = self._parse_basic_yaml(content)

            # Verify configuration structure
            if not isinstance(config_data, dict):
                result.add_error("Configuration root must be a dictionary")
                return False

            # Check for required sections
            missing_sections = []
            existing_sections = []

            for section in self.required_sections:
                if section in config_data:
                    existing_sections.append(section)
                else:
                    missing_sections.append(section)

            if missing_sections:
                result.add_warning(
                    f"Missing recommended sections: {missing_sections}"
                )

            result.add_metadata("existing_sections", existing_sections)
            result.add_metadata("missing_sections", missing_sections)
            result.add_metadata("config_keys", list(config_data.keys()))

            # Verify section structures
            self._verify_section_structures(config_data, result)

            return True

        except json.JSONDecodeError as e:
            result.add_error(f"Invalid JSON format: {e}")
            return False
        except Exception as e:
            result.add_error(f"Failed to parse configuration: {e}")
            return False

    def _parse_basic_yaml(self, content: str) -> dict[str, Any]:
        """Basic YAML parsing for structure validation."""
        # This is a simplified YAML parser for basic structure validation
        # In a real implementation, you would use PyYAML
        config_data = {}
        current_section = None

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
                        config_data[current_section][key] = value
                    else:
                        config_data[key] = value

        return config_data

    def _verify_section_structures(
        self, config_data: dict[str, Any], result: VerificationResult
    ) -> None:
        """Verify the structure of configuration sections."""
        try:
            # Verify model section
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
                        result.add_warning(
                            "Missing recommended model fields: "
                            f"{missing_model_fields}"
                        )

                    result.add_metadata(
                        "model_fields", list(model_config.keys())
                    )

            # Verify training section
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
                        result.add_warning(
                            "Missing recommended training fields: "
                            f"{missing_training_fields}"
                        )

                    result.add_metadata(
                        "training_fields", list(training_config.keys())
                    )

            # Verify data section
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
                        result.add_warning(
                            "Missing recommended data fields: "
                            f"{missing_data_fields}"
                        )

                    result.add_metadata(
                        "data_fields", list(data_config.keys())
                    )

            # Verify experiment section
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
                        result.add_warning(
                            "Missing recommended experiment fields: "
                            f"{missing_experiment_fields}"
                        )

                    result.add_metadata(
                        "experiment_fields", list(experiment_config.keys())
                    )

        except Exception as e:
            result.add_warning(f"Section structure verification failed: {e}")

    def _verify_config_structure(
        self, config_path: Path, result: VerificationResult
    ) -> None:
        """Perform deep configuration structure analysis."""
        try:
            file_extension = config_path.suffix.lower()

            if file_extension == ".json":
                with open(config_path, encoding="utf-8") as f:
                    config_data = json.load(f)

                # Analyze configuration structure
                result.add_metadata(
                    "max_depth", self._calculate_dict_depth(config_data)
                )
                result.add_metadata(
                    "total_keys", self._count_dict_keys(config_data)
                )

                # Check for nested configurations
                nested_configs = self._find_nested_configs(config_data)
                if nested_configs:
                    result.add_metadata("nested_configs", nested_configs)

                # Check for environment variables
                env_vars = self._find_environment_variables(config_data)
                if env_vars:
                    result.add_metadata("environment_variables", env_vars)

            else:
                # For YAML files, basic structure analysis
                with open(config_path, encoding="utf-8") as f:
                    content = f.read()

                lines = content.splitlines()
                result.add_metadata("config_lines", len(lines))
                result.add_metadata("config_size", len(content))

                # Check for common patterns
                if "hydra:" in content.lower():
                    result.add_metadata("has_hydra_config", True)

                if "defaults:" in content.lower():
                    result.add_metadata("has_defaults", True)

        except Exception as e:
            result.add_warning(f"Deep configuration analysis failed: {e}")

    def _verify_config_consistency(
        self, config_path: Path, result: VerificationResult
    ) -> None:
        """Verify configuration consistency and cross-references."""
        try:
            file_extension = config_path.suffix.lower()

            if file_extension == ".json":
                with open(config_path, encoding="utf-8") as f:
                    config_data = json.load(f)

                # Check for circular references
                circular_refs = self._check_circular_references(config_data)
                if circular_refs:
                    result.add_warning(
                        f"Potential circular references: {circular_refs}"
                    )

                # Check for unused imports or references
                unused_refs = self._check_unused_references(config_data)
                if unused_refs:
                    result.add_warning(
                        f"Potential unused references: {unused_refs}"
                    )

                # Verify path consistency
                path_issues = self._verify_path_consistency(config_data)
                if path_issues:
                    result.add_warning(
                        f"Path consistency issues: {path_issues}"
                    )

        except Exception as e:
            result.add_warning(f"Configuration consistency check failed: {e}")

    def _calculate_dict_depth(
        self, data: dict[str, Any], current_depth: int = 0
    ) -> int:
        """Calculate the maximum depth of a nested dictionary."""
        if not isinstance(data, dict):
            return current_depth

        max_depth = current_depth
        for value in data.values():
            if isinstance(value, dict):
                depth = self._calculate_dict_depth(value, current_depth + 1)
                max_depth = max(max_depth, depth)

        return max_depth

    def _count_dict_keys(self, data: dict[str, Any]) -> int:
        """Count the total number of keys in a nested dictionary."""
        if not isinstance(data, dict):
            return 0

        total_keys = len(data)
        for value in data.values():
            if isinstance(value, dict):
                total_keys += self._count_dict_keys(value)

        return total_keys

    def _find_nested_configs(self, config_data: dict[str, Any]) -> list[str]:
        """Find nested configuration references."""
        nested_configs = []

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

    def _find_environment_variables(
        self, config_data: dict[str, Any]
    ) -> list[str]:
        """Find environment variable references."""
        env_vars = []

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

    def _check_circular_references(
        self, config_data: dict[str, Any]
    ) -> list[str]:
        """Check for potential circular references."""
        # This is a simplified check - in practice, you'd need more
        # sophisticated analysis
        circular_refs = []

        def _check_refs(data: Any, visited: set, path: str = "") -> None:
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

    def _check_unused_references(
        self, config_data: dict[str, Any]
    ) -> list[str]:
        """Check for potentially unused references."""
        # This is a simplified check - in practice, you'd need more
        # sophisticated analysis
        unused_refs = []

        # Check for common patterns that might indicate unused references
        if isinstance(config_data, dict):
            for _key, value in config_data.items():
                if (
                    isinstance(value, str)
                    and value.startswith("${")
                    and value.endswith("}")
                ):
                    # This is a placeholder for more sophisticated analysis
                    pass

        return unused_refs

    def _verify_path_consistency(
        self, config_data: dict[str, Any]
    ) -> list[str]:
        """Verify path consistency in configuration."""
        path_issues = []

        def _check_paths(data: Any, path: str = "") -> None:
            if isinstance(data, dict):
                for key, value in data.items():
                    current_path = f"{path}.{key}" if path else key
                    if isinstance(value, str) and any(
                        keyword in key.lower()
                        for keyword in ["path", "dir", "file"]
                    ):
                        # Check if path is relative and might be problematic
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
