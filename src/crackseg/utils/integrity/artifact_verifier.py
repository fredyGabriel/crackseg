"""
Artifact integrity verification.

This module provides specialized integrity verification for general artifacts,
integrating with the existing ArtifactManager and ArtifactVersioner systems.
"""

import json
import logging
from pathlib import Path
from typing import Any

from .core import IntegrityVerifier, VerificationLevel, VerificationResult

logger = logging.getLogger(__name__)


class ArtifactIntegrityVerifier(IntegrityVerifier):
    """Specialized verifier for general artifact integrity."""

    def __init__(
        self,
        verification_level: VerificationLevel = VerificationLevel.STANDARD,
        supported_extensions: list[str] | None = None,
    ):
        """
        Initialize artifact integrity verifier.

        Args:
            verification_level: Level of verification thoroughness
            supported_extensions: List of supported file extensions
        """
        super().__init__(verification_level)
        self.supported_extensions = supported_extensions or [
            ".pth",
            ".pt",
            ".ckpt",
            ".json",
            ".yaml",
            ".yml",
            ".csv",
            ".txt",
            ".log",
            ".png",
            ".jpg",
            ".jpeg",
        ]

    def verify(self, artifact_path: Path) -> VerificationResult:
        """
        Verify the integrity of a general artifact.

        Args:
            artifact_path: Path to the artifact file

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
        if not self._verify_file_extension(artifact_path, result):
            return result

        if self.verification_level in [
            VerificationLevel.STANDARD,
            VerificationLevel.THOROUGH,
            VerificationLevel.PARANOID,
        ]:
            # Content validation based on file type
            if not self._verify_content_by_type(artifact_path, result):
                return result

        if self.verification_level in [
            VerificationLevel.THOROUGH,
            VerificationLevel.PARANOID,
        ]:
            # Deep content analysis
            self._verify_content_structure(artifact_path, result)

        if self.verification_level == VerificationLevel.PARANOID:
            # Cross-reference validation
            self._verify_metadata_consistency(artifact_path, result)

        return result

    def _verify_file_extension(
        self, artifact_path: Path, result: VerificationResult
    ) -> bool:
        """Verify that the file has a supported extension."""
        file_extension = artifact_path.suffix.lower()

        if file_extension not in self.supported_extensions:
            result.add_warning(f"Unsupported file extension: {file_extension}")
            # Don't fail for unsupported extensions, just warn

        result.add_metadata("file_extension", file_extension)
        return True

    def _verify_content_by_type(
        self, artifact_path: Path, result: VerificationResult
    ) -> bool:
        """Verify content based on file type."""
        file_extension = artifact_path.suffix.lower()

        if file_extension in [".json", ".yaml", ".yml"]:
            return self._verify_structured_data(artifact_path, result)
        elif file_extension in [".csv", ".txt", ".log"]:
            return self._verify_text_data(artifact_path, result)
        elif file_extension in [".png", ".jpg", ".jpeg"]:
            return self._verify_image_data(artifact_path, result)
        else:
            # For other file types, just verify they can be read
            return self._verify_binary_data(artifact_path, result)

    def _verify_structured_data(
        self, artifact_path: Path, result: VerificationResult
    ) -> bool:
        """Verify JSON/YAML structured data files."""
        try:
            with open(artifact_path, encoding="utf-8") as f:
                if artifact_path.suffix.lower() == ".json":
                    data = json.load(f)
                else:
                    # For YAML files, we'll just check if they can be read
                    # Full YAML parsing would require PyYAML dependency
                    content = f.read()
                    if not content.strip():
                        result.add_warning("YAML file is empty")
                        return True
                    data = {"content": content}  # Placeholder for YAML

            if isinstance(data, dict):
                result.add_metadata("data_keys", list(data.keys()))
                result.add_metadata("data_type", "structured")
            elif isinstance(data, list):
                result.add_metadata("data_length", len(data))
                result.add_metadata("data_type", "list")
            else:
                result.add_metadata("data_type", type(data).__name__)

            return True

        except json.JSONDecodeError as e:
            result.add_error(f"Invalid JSON format: {e}")
            return False
        except Exception as e:
            result.add_error(f"Failed to parse structured data: {e}")
            return False

    def _verify_text_data(
        self, artifact_path: Path, result: VerificationResult
    ) -> bool:
        """Verify text-based data files."""
        try:
            with open(artifact_path, encoding="utf-8") as f:
                content = f.read()

            lines = content.splitlines()
            result.add_metadata("line_count", len(lines))
            result.add_metadata("character_count", len(content))
            result.add_metadata("data_type", "text")

            # Check for common issues
            if not content.strip():
                result.add_warning("Text file is empty")

            if len(lines) > 0 and not lines[0].strip():
                result.add_warning("Text file starts with empty line")

            if len(lines) > 0 and not lines[-1].strip():
                result.add_warning("Text file ends with empty line")

            return True

        except UnicodeDecodeError as e:
            result.add_error(f"Text file encoding error: {e}")
            return False
        except Exception as e:
            result.add_error(f"Failed to read text file: {e}")
            return False

    def _verify_image_data(
        self, artifact_path: Path, result: VerificationResult
    ) -> bool:
        """Verify image data files."""
        try:
            # Basic image file validation
            with open(artifact_path, "rb") as f:
                header = f.read(8)

            # Check for common image format signatures
            png_signature = b"\x89PNG\r\n\x1a\n"
            jpeg_signatures = [b"\xff\xd8\xff"]

            if header.startswith(png_signature):
                result.add_metadata("image_format", "PNG")
            elif any(header.startswith(sig) for sig in jpeg_signatures):
                result.add_metadata("image_format", "JPEG")
            else:
                result.add_warning("Unknown image format")
                result.add_metadata("image_format", "unknown")

            result.add_metadata("data_type", "image")
            return True

        except Exception as e:
            result.add_error(f"Failed to verify image data: {e}")
            return False

    def _verify_binary_data(
        self, artifact_path: Path, result: VerificationResult
    ) -> bool:
        """Verify binary data files."""
        try:
            with open(artifact_path, "rb") as f:
                # Read first few bytes to check if file is accessible
                header = f.read(16)

            result.add_metadata("data_type", "binary")
            result.add_metadata("header_hex", header.hex())
            return True

        except Exception as e:
            result.add_error(f"Failed to read binary file: {e}")
            return False

    def _verify_content_structure(
        self, artifact_path: Path, result: VerificationResult
    ) -> None:
        """Perform deep content structure analysis."""
        try:
            file_extension = artifact_path.suffix.lower()

            if file_extension == ".json":
                with open(artifact_path, encoding="utf-8") as f:
                    data = json.load(f)

                # Analyze JSON structure
                if isinstance(data, dict):
                    result.add_metadata(
                        "max_depth", self._calculate_dict_depth(data)
                    )
                    result.add_metadata(
                        "total_keys", self._count_dict_keys(data)
                    )
                elif isinstance(data, list):
                    result.add_metadata("list_length", len(data))
                    if data:
                        result.add_metadata(
                            "list_item_types",
                            [type(item).__name__ for item in data[:5]],
                        )

            elif file_extension in [".csv", ".txt", ".log"]:
                with open(artifact_path, encoding="utf-8") as f:
                    lines = f.readlines()

                # Analyze text structure
                non_empty_lines = [line for line in lines if line.strip()]
                result.add_metadata("non_empty_lines", len(non_empty_lines))
                result.add_metadata(
                    "empty_lines", len(lines) - len(non_empty_lines)
                )

                if lines:
                    avg_line_length = sum(len(line) for line in lines) / len(
                        lines
                    )
                    result.add_metadata(
                        "average_line_length", round(avg_line_length, 2)
                    )

        except Exception as e:
            result.add_warning(f"Deep content analysis failed: {e}")

    def _verify_metadata_consistency(
        self, artifact_path: Path, result: VerificationResult
    ) -> None:
        """Verify metadata consistency and cross-references."""
        try:
            # Check for associated metadata files
            metadata_path = artifact_path.with_suffix(
                artifact_path.suffix + ".meta"
            )
            if metadata_path.exists():
                result.add_metadata("has_metadata_file", True)

                # Verify metadata file integrity
                if metadata_path.suffix.lower() == ".json":
                    with open(metadata_path, encoding="utf-8") as f:
                        metadata = json.load(f)

                    # Check if metadata references the main file
                    if "original_file" in metadata:
                        original_file = metadata["original_file"]
                        if Path(original_file).name != artifact_path.name:
                            result.add_warning(
                                "Metadata file references different original "
                                "file"
                            )

                    # Check checksum consistency if available
                    if "checksum" in metadata:
                        expected_checksum = metadata["checksum"]
                        if (
                            result.checksum
                            and result.checksum != expected_checksum
                        ):
                            result.add_error(
                                "Checksum mismatch with metadata file"
                            )
            else:
                result.add_metadata("has_metadata_file", False)

        except Exception as e:
            result.add_warning(f"Metadata consistency check failed: {e}")

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
