"""
Experiment integrity verification.

This module provides specialized integrity verification for experiment data,
integrating with the ExperimentTracker system.
"""

import json
import logging
from pathlib import Path

from .core import IntegrityVerifier, VerificationLevel, VerificationResult

logger = logging.getLogger(__name__)


class ExperimentIntegrityVerifier(IntegrityVerifier):
    """Specialized verifier for experiment data integrity."""

    def __init__(
        self,
        verification_level: VerificationLevel = VerificationLevel.STANDARD,
        required_files: list[str] | None = None,
    ):
        """
        Initialize experiment integrity verifier.

        Args:
            verification_level: Level of verification thoroughness
            required_files: List of required files in experiment directory
        """
        super().__init__(verification_level)
        self.required_files = required_files or [
            "experiment_tracker.json",
            "config.yaml",
            "metrics.jsonl",
        ]

    def verify(self, artifact_path: Path) -> VerificationResult:
        """
        Verify the integrity of an experiment directory.

        Args:
            artifact_path: Path to the experiment directory

        Returns:
            VerificationResult with verification details
        """
        result = VerificationResult(
            is_valid=True,
            artifact_path=artifact_path,
            verification_level=self.verification_level,
        )

        # Verify it's a directory
        if not artifact_path.is_dir():
            result.add_error(f"Not a directory: {artifact_path}")
            return result

        # Basic structure verification
        if not self._verify_experiment_structure(artifact_path, result):
            return result

        if self.verification_level in [
            VerificationLevel.STANDARD,
            VerificationLevel.THOROUGH,
            VerificationLevel.PARANOID,
        ]:
            # Verify experiment metadata
            if not self._verify_experiment_metadata(artifact_path, result):
                return result

        if self.verification_level in [
            VerificationLevel.THOROUGH,
            VerificationLevel.PARANOID,
        ]:
            # Deep content analysis
            self._verify_experiment_content(artifact_path, result)

        if self.verification_level == VerificationLevel.PARANOID:
            # Cross-reference validation
            self._verify_experiment_consistency(artifact_path, result)

        return result

    def _verify_experiment_structure(
        self, experiment_path: Path, result: VerificationResult
    ) -> bool:
        """Verify experiment directory structure."""
        # Check for required files
        missing_files = []
        existing_files = []

        for required_file in self.required_files:
            file_path = experiment_path / required_file
            if file_path.exists():
                existing_files.append(required_file)
            else:
                missing_files.append(required_file)

        if missing_files:
            result.add_warning(f"Missing recommended files: {missing_files}")

        result.add_metadata("existing_files", existing_files)
        result.add_metadata("missing_files", missing_files)
        result.add_metadata("total_files", len(existing_files))

        # Check for common experiment subdirectories
        subdirs = [d for d in experiment_path.iterdir() if d.is_dir()]
        result.add_metadata("subdirectories", [d.name for d in subdirs])

        return True

    def _verify_experiment_metadata(
        self, experiment_path: Path, result: VerificationResult
    ) -> bool:
        """Verify experiment metadata files."""
        metadata_file = experiment_path / "experiment_tracker.json"

        if not metadata_file.exists():
            result.add_warning("Experiment metadata file not found")
            return True  # Don't fail, just warn

        try:
            with open(metadata_file, encoding="utf-8") as f:
                metadata = json.load(f)

            # Verify metadata structure
            if not isinstance(metadata, dict):
                result.add_error("Experiment metadata is not a dictionary")
                return False

            # Check for required metadata fields
            required_metadata_fields = [
                "experiment_id",
                "created_at",
                "status",
            ]
            missing_metadata_fields = []

            for field in required_metadata_fields:
                if field not in metadata:
                    missing_metadata_fields.append(field)

            if missing_metadata_fields:
                result.add_warning(
                    f"Missing metadata fields: {missing_metadata_fields}"
                )

            # Add metadata to result
            result.add_metadata("experiment_id", metadata.get("experiment_id"))
            result.add_metadata("experiment_status", metadata.get("status"))
            result.add_metadata("created_at", metadata.get("created_at"))
            result.add_metadata("metadata_keys", list(metadata.keys()))

            return True

        except json.JSONDecodeError as e:
            result.add_error(f"Invalid JSON in experiment metadata: {e}")
            return False
        except Exception as e:
            result.add_error(f"Failed to read experiment metadata: {e}")
            return False

    def _verify_experiment_content(
        self, experiment_path: Path, result: VerificationResult
    ) -> None:
        """Perform deep content analysis of experiment files."""
        try:
            # Verify config file
            config_file = experiment_path / "config.yaml"
            if config_file.exists():
                self._verify_config_file(config_file, result)

            # Verify metrics file
            metrics_file = experiment_path / "metrics.jsonl"
            if metrics_file.exists():
                self._verify_metrics_file(metrics_file, result)

            # Check for checkpoint files
            checkpoint_files = list(experiment_path.glob("*.pth")) + list(
                experiment_path.glob("*.ckpt")
            )
            if checkpoint_files:
                result.add_metadata("checkpoint_count", len(checkpoint_files))
                result.add_metadata(
                    "checkpoint_files", [f.name for f in checkpoint_files]
                )

            # Check for visualization files
            viz_files = list(experiment_path.glob("*.png")) + list(
                experiment_path.glob("*.jpg")
            )
            if viz_files:
                result.add_metadata("visualization_count", len(viz_files))
                result.add_metadata(
                    "visualization_files", [f.name for f in viz_files]
                )

        except Exception as e:
            result.add_warning(f"Deep content analysis failed: {e}")

    def _verify_config_file(
        self, config_file: Path, result: VerificationResult
    ) -> None:
        """Verify experiment configuration file."""
        try:
            with open(config_file, encoding="utf-8") as f:
                content = f.read()

            # Basic YAML validation (just check if it can be read)
            if not content.strip():
                result.add_warning("Configuration file is empty")
            else:
                result.add_metadata("config_file_size", len(content))
                result.add_metadata("config_lines", len(content.splitlines()))

                # Check for common configuration sections
                config_sections = ["model", "training", "data", "experiment"]
                found_sections = []

                for section in config_sections:
                    if section in content.lower():
                        found_sections.append(section)

                result.add_metadata("config_sections", found_sections)

        except Exception as e:
            result.add_warning(f"Failed to verify config file: {e}")

    def _verify_metrics_file(
        self, metrics_file: Path, result: VerificationResult
    ) -> None:
        """Verify experiment metrics file."""
        try:
            with open(metrics_file, encoding="utf-8") as f:
                lines = f.readlines()

            if not lines:
                result.add_warning("Metrics file is empty")
                return

            # Parse JSONL format
            valid_metrics = 0
            invalid_metrics = 0
            metric_keys = set()

            for _line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    metric_data = json.loads(line)
                    if isinstance(metric_data, dict):
                        valid_metrics += 1
                        metric_keys.update(metric_data.keys())
                    else:
                        invalid_metrics += 1
                except json.JSONDecodeError:
                    invalid_metrics += 1

            result.add_metadata("total_metric_lines", len(lines))
            result.add_metadata("valid_metrics", valid_metrics)
            result.add_metadata("invalid_metrics", invalid_metrics)
            result.add_metadata("metric_keys", list(metric_keys))

            if invalid_metrics > 0:
                result.add_warning(
                    f"Found {invalid_metrics} invalid metric entries"
                )

        except Exception as e:
            result.add_warning(f"Failed to verify metrics file: {e}")

    def _verify_experiment_consistency(
        self, experiment_path: Path, result: VerificationResult
    ) -> None:
        """Verify experiment consistency and cross-references."""
        try:
            # Check metadata file
            metadata_file = experiment_path / "experiment_tracker.json"
            if metadata_file.exists():
                with open(metadata_file, encoding="utf-8") as f:
                    metadata = json.load(f)

                # Verify experiment ID consistency
                experiment_id = metadata.get("experiment_id")
                if experiment_id:
                    # Check if experiment ID matches directory name
                    dir_name = experiment_path.name
                    if experiment_id not in dir_name:
                        result.add_warning(
                            f"Experiment ID '{experiment_id}' not found in "
                            "directory name "
                            f"'{dir_name}'"
                        )

                # Check for referenced files
                if "artifacts" in metadata:
                    artifacts = metadata["artifacts"]
                    if isinstance(artifacts, list):
                        missing_artifacts = []
                        for artifact in artifacts:
                            if (
                                isinstance(artifact, dict)
                                and "path" in artifact
                            ):
                                artifact_path = (
                                    experiment_path / artifact["path"]
                                )
                                if not artifact_path.exists():
                                    missing_artifacts.append(artifact["path"])

                        if missing_artifacts:
                            result.add_warning(
                                "Referenced artifacts not found: "
                                f"{missing_artifacts}"
                            )

            # Check for orphaned files (files not referenced in metadata)
            all_files = [
                f.name for f in experiment_path.iterdir() if f.is_file()
            ]
            result.add_metadata("total_files", len(all_files))

        except Exception as e:
            result.add_warning(f"Consistency verification failed: {e}")
