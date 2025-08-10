"""
Configuration integrity verification.

This module provides specialized integrity verification for configuration
files, integrating with the existing configuration validation system.
"""

import json
import logging
from pathlib import Path

from .config_verifier_helpers import (
    analyze_yaml_text,
    cross_check_json,
    deep_analyze_json,
    load_config_content,
    verify_extension,
    verify_required_sections,
    verify_structures,
)
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
        return verify_extension(
            config_path, result.add_metadata, result.add_error
        )

    def _verify_config_content(
        self, config_path: Path, result: VerificationResult
    ) -> bool:
        """Verify configuration content and structure."""
        try:
            config_data = load_config_content(config_path)

            # Verify configuration structure
            if not isinstance(config_data, dict):
                result.add_error("Configuration root must be a dictionary")
                return False

            # Check for required sections
            existing_sections, missing_sections = verify_required_sections(
                config_data, self.required_sections
            )

            if missing_sections:
                result.add_warning(
                    f"Missing recommended sections: {missing_sections}"
                )

            result.add_metadata("existing_sections", existing_sections)
            result.add_metadata("missing_sections", missing_sections)
            result.add_metadata("config_keys", list(config_data.keys()))

            # Verify section structures
            verify_structures(
                config_data, result.add_warning, result.add_metadata
            )

            return True

        except json.JSONDecodeError as e:
            result.add_error(f"Invalid JSON format: {e}")
            return False
        except Exception as e:
            result.add_error(f"Failed to parse configuration: {e}")
            return False

    def _verify_config_structure(
        self, config_path: Path, result: VerificationResult
    ) -> None:
        """Perform deep configuration structure analysis."""
        try:
            file_extension = config_path.suffix.lower()
            if file_extension == ".json":
                with open(config_path, encoding="utf-8") as f:
                    config_data = json.load(f)
                for k, v in deep_analyze_json(config_data).items():
                    result.add_metadata(k, v)
            else:
                with open(config_path, encoding="utf-8") as f:
                    content = f.read()
                for k, v in analyze_yaml_text(content).items():
                    result.add_metadata(k, v)

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
                meta = cross_check_json(config_data)
                if "circular_references" in meta:
                    result.add_warning(
                        f"Potential circular references: {meta['circular_references']}"
                    )
                if "unused_references" in meta:
                    result.add_warning(
                        f"Potential unused references: {meta['unused_references']}"
                    )
                if "path_issues" in meta:
                    result.add_warning(
                        f"Path consistency issues: {meta['path_issues']}"
                    )

        except Exception as e:
            result.add_warning(f"Configuration consistency check failed: {e}")
