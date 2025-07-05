"""Storage management for capture artifacts.

This module provides storage utilities for managing screenshots, videos,
and visual regression artifacts with configurable retention policies,
naming conventions, and automatic cleanup capabilities.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class RetentionPolicy(Enum):
    """Retention policies for captured artifacts."""

    KEEP_ALL = "keep_all"
    KEEP_FAILURES = "keep_failures"
    KEEP_RECENT = "keep_recent"
    KEEP_NONE = "keep_none"


class NamingConvention(Enum):
    """Naming conventions for captured files."""

    TIMESTAMP = "timestamp"  # test_20240115_143052.png
    TEST_NAME = "test_name"  # test_login_page.png
    SEQUENTIAL = "sequential"  # test_001.png
    DESCRIPTIVE = "descriptive"  # test_login_page_failure_20240115.png


@dataclass
class StorageConfig:
    """Configuration for capture storage management.

    Attributes:
        base_dir: Base directory for all capture artifacts
        screenshots_dir: Subdirectory for screenshots
        videos_dir: Subdirectory for videos
        regression_dir: Subdirectory for visual regression artifacts
        temp_dir: Temporary directory for processing
        retention_policy: Policy for cleaning up old artifacts
        retention_days: Days to keep artifacts (for KEEP_RECENT policy)
        max_files_per_test: Maximum files per test (0 = unlimited)
        naming_convention: File naming convention
        compress_old_files: Compress files older than threshold
        compression_threshold_days: Days after which to compress files
        auto_cleanup: Enable automatic cleanup on startup
        preserve_failure_artifacts: Always preserve artifacts from failed tests
    """

    base_dir: Path = field(default_factory=lambda: Path("test-artifacts"))
    screenshots_dir: str = "screenshots"
    videos_dir: str = "videos"
    regression_dir: str = "visual-regression"
    temp_dir: str = "temp"
    retention_policy: RetentionPolicy = RetentionPolicy.KEEP_FAILURES
    retention_days: int = 7
    max_files_per_test: int = 10
    naming_convention: NamingConvention = NamingConvention.DESCRIPTIVE
    compress_old_files: bool = True
    compression_threshold_days: int = 3
    auto_cleanup: bool = True
    preserve_failure_artifacts: bool = True


class HasCaptureStorage(Protocol):
    """Protocol for classes that support capture storage management."""

    def store_artifact(
        self,
        file_path: Path,
        artifact_type: str,
        test_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Store an artifact with proper naming and organization."""
        ...

    def cleanup_artifacts(
        self,
        test_name: str | None = None,
        force: bool = False,
    ) -> int:
        """Clean up old artifacts according to retention policy."""
        ...


class CaptureStorage:
    """Storage management for capture artifacts.

    Handles file organization, retention policies, cleanup, and compression
    for screenshots, videos, and visual regression artifacts.
    """

    def __init__(self, config: StorageConfig | None = None) -> None:
        """Initialize storage management with configuration.

        Args:
            config: Storage configuration
        """
        self.config = config or StorageConfig()
        self._setup_directories()

        if self.config.auto_cleanup:
            self._auto_cleanup()

        logger.debug(f"Storage initialized: {self.config.base_dir}")

    def store_artifact(
        self,
        file_path: Path,
        artifact_type: str,
        test_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Store an artifact with proper naming and organization.

        Args:
            file_path: Path to the source file
            artifact_type: Type of artifact (screenshot, video, regression)
            test_name: Name of the test that generated this artifact
            metadata: Optional metadata to store with the artifact

        Returns:
            Path to the stored artifact

        Raises:
            FileNotFoundError: If source file doesn't exist
            OSError: If storage operation fails
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Source file not found: {file_path}")

        try:
            # Determine target directory
            target_dir = self._get_target_directory(artifact_type)

            # Generate target filename
            target_name = self._generate_filename(
                test_name, artifact_type, file_path.suffix, metadata
            )

            target_path = target_dir / target_name

            # Ensure unique filename
            target_path = self._ensure_unique_path(target_path)

            # Copy file to target location
            if file_path != target_path:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                import shutil

                shutil.copy2(file_path, target_path)

                logger.debug(f"Stored artifact: {target_path}")

            # Store metadata if provided
            if metadata:
                self._store_metadata(target_path, metadata)

            # Apply retention policy
            self._apply_retention_policy(target_dir, test_name)

            return target_path

        except Exception as e:
            logger.error(f"Failed to store artifact {file_path}: {e}")
            raise OSError(f"Storage operation failed: {e}") from e

    def cleanup_artifacts(
        self,
        test_name: str | None = None,
        force: bool = False,
    ) -> int:
        """Clean up old artifacts according to retention policy.

        Args:
            test_name: Specific test to clean (None for all tests)
            force: Force cleanup regardless of retention policy

        Returns:
            Number of files cleaned up
        """
        cleaned_count = 0

        try:
            if test_name:
                # Clean specific test artifacts
                cleaned_count += self._cleanup_test_artifacts(test_name, force)
            else:
                # Clean all artifacts
                for artifact_dir in self._get_all_artifact_directories():
                    if artifact_dir.exists():
                        cleaned_count += self._cleanup_directory(
                            artifact_dir, force
                        )

            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} artifact files")

            return cleaned_count

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return 0

    def get_artifact_info(self, test_name: str) -> dict[str, Any]:
        """Get information about artifacts for a specific test.

        Args:
            test_name: Name of the test

        Returns:
            Dictionary with artifact information
        """
        info: dict[str, Any] = {
            "test_name": test_name,
            "artifacts": {},
            "total_files": 0,
            "total_size": 0,
            "last_modified": None,
        }

        total_files = 0
        total_size = 0
        last_modified: datetime | None = None

        for artifact_type in ["screenshot", "video", "regression"]:
            artifact_dir = self._get_target_directory(artifact_type)
            pattern = f"*{test_name}*"
            files = list(artifact_dir.glob(pattern))

            count = len(files)
            file_list = [f.name for f in files]
            size = sum(f.stat().st_size for f in files if f.is_file())

            artifact_info = {
                "count": count,
                "files": file_list,
                "size": size,
            }

            info["artifacts"][artifact_type] = artifact_info
            total_files += count
            total_size += size

            # Update last modified
            for file_path in files:
                if file_path.is_file():
                    file_mtime = datetime.fromtimestamp(
                        file_path.stat().st_mtime
                    )
                    if last_modified is None or file_mtime > last_modified:
                        last_modified = file_mtime

        info["total_files"] = total_files
        info["total_size"] = total_size
        info["last_modified"] = last_modified
        return info

    def _setup_directories(self) -> None:
        """Create necessary directories for storage."""
        dirs_to_create = [
            self.config.base_dir,
            self.config.base_dir / self.config.screenshots_dir,
            self.config.base_dir / self.config.videos_dir,
            self.config.base_dir / self.config.regression_dir,
            self.config.base_dir / self.config.temp_dir,
        ]

        for directory in dirs_to_create:
            directory.mkdir(parents=True, exist_ok=True)

    def _get_target_directory(self, artifact_type: str) -> Path:
        """Get target directory for artifact type."""
        type_mapping = {
            "screenshot": self.config.screenshots_dir,
            "video": self.config.videos_dir,
            "regression": self.config.regression_dir,
        }

        subdir = type_mapping.get(artifact_type, artifact_type)
        return self.config.base_dir / subdir

    def _generate_filename(
        self,
        test_name: str,
        artifact_type: str,
        extension: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Generate filename based on naming convention."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_test_name = self._sanitize_filename(test_name)

        if self.config.naming_convention == NamingConvention.TIMESTAMP:
            return f"{clean_test_name}_{timestamp}{extension}"

        elif self.config.naming_convention == NamingConvention.TEST_NAME:
            return f"{clean_test_name}_{artifact_type}{extension}"

        elif self.config.naming_convention == NamingConvention.SEQUENTIAL:
            seq_num = self._get_next_sequence_number(test_name, artifact_type)
            return f"{clean_test_name}_{seq_num:03d}{extension}"

        elif self.config.naming_convention == NamingConvention.DESCRIPTIVE:
            status = (
                "failure"
                if metadata and metadata.get("test_failed")
                else "success"
            )
            return f"{clean_test_name}_{status}_{timestamp}{extension}"

        else:
            return f"{clean_test_name}_{timestamp}{extension}"

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to remove invalid characters."""
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, "_")
        return filename[:100]  # Limit length

    def _ensure_unique_path(self, path: Path) -> Path:
        """Ensure the path is unique by adding suffix if needed."""
        if not path.exists():
            return path

        base = path.stem
        extension = path.suffix
        parent = path.parent
        counter = 1

        while True:
            new_name = f"{base}_{counter:02d}{extension}"
            new_path = parent / new_name
            if not new_path.exists():
                return new_path
            counter += 1

    def _get_next_sequence_number(
        self, test_name: str, artifact_type: str
    ) -> int:
        """Get next sequence number for a test."""
        target_dir = self._get_target_directory(artifact_type)
        clean_test_name = self._sanitize_filename(test_name)
        pattern = f"{clean_test_name}_*..*"
        existing_files = list(target_dir.glob(pattern))

        if not existing_files:
            return 1

        max_num = 0
        for file_path in existing_files:
            try:
                # Extract number from filename like "test_001.png"
                parts = file_path.stem.split("_")
                if len(parts) >= 2 and parts[-1].isdigit():
                    num = int(parts[-1])
                    max_num = max(max_num, num)
            except (ValueError, IndexError):
                continue

        return max_num + 1

    def _store_metadata(
        self, file_path: Path, metadata: dict[str, Any]
    ) -> None:
        """Store metadata alongside artifact file."""
        try:
            import json

            metadata_path = file_path.with_suffix(file_path.suffix + ".meta")
            metadata_with_timestamp = {
                **metadata,
                "stored_at": datetime.now().isoformat(),
                "original_path": str(file_path),
            }

            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata_with_timestamp, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to store metadata for {file_path}: {e}")

    def _apply_retention_policy(self, directory: Path, test_name: str) -> None:
        """Apply retention policy to directory."""
        if self.config.retention_policy == RetentionPolicy.KEEP_ALL:
            return

        try:
            if self.config.max_files_per_test > 0:
                self._enforce_file_limit(directory, test_name)

            if self.config.retention_policy == RetentionPolicy.KEEP_RECENT:
                self._cleanup_old_files(directory)

        except Exception as e:
            logger.warning(f"Failed to apply retention policy: {e}")

    def _enforce_file_limit(self, directory: Path, test_name: str) -> None:
        """Enforce maximum file limit per test."""
        clean_test_name = self._sanitize_filename(test_name)
        pattern = f"*{clean_test_name}*"
        files = sorted(
            directory.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        files_to_remove = files[self.config.max_files_per_test :]
        for file_path in files_to_remove:
            try:
                if file_path.is_file():
                    file_path.unlink()
                    # Also remove metadata file if exists
                    meta_file = file_path.with_suffix(
                        file_path.suffix + ".meta"
                    )
                    if meta_file.exists():
                        meta_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove old file {file_path}: {e}")

    def _cleanup_old_files(self, directory: Path) -> None:
        """Clean up files older than retention period."""
        cutoff_time = time.time() - (self.config.retention_days * 24 * 3600)

        for file_path in directory.iterdir():
            if not file_path.is_file():
                continue

            try:
                if file_path.stat().st_mtime < cutoff_time:
                    # Check if failure artifact should be preserved
                    if self.config.preserve_failure_artifacts:
                        meta_file = file_path.with_suffix(
                            file_path.suffix + ".meta"
                        )
                        if meta_file.exists():
                            import json

                            with open(meta_file, encoding="utf-8") as f:
                                metadata = json.load(f)
                                if metadata.get("test_failed", False):
                                    continue  # Preserve failure artifacts

                    file_path.unlink()
                    # Also remove metadata file
                    meta_file = file_path.with_suffix(
                        file_path.suffix + ".meta"
                    )
                    if meta_file.exists():
                        meta_file.unlink()

            except Exception as e:
                logger.warning(f"Failed to clean up old file {file_path}: {e}")

    def _cleanup_test_artifacts(self, test_name: str, force: bool) -> int:
        """Clean up artifacts for a specific test."""
        cleaned_count = 0
        clean_test_name = self._sanitize_filename(test_name)

        for artifact_dir in self._get_all_artifact_directories():
            if not artifact_dir.exists():
                continue

            pattern = f"*{clean_test_name}*"
            files = list(artifact_dir.glob(pattern))

            for file_path in files:
                try:
                    if force or self._should_cleanup_file(file_path):
                        if file_path.is_file():
                            file_path.unlink()
                            cleaned_count += 1

                            # Also remove metadata file
                            meta_file = file_path.with_suffix(
                                file_path.suffix + ".meta"
                            )
                            if meta_file.exists():
                                meta_file.unlink()

                except Exception as e:
                    logger.warning(f"Failed to clean up {file_path}: {e}")

        return cleaned_count

    def _cleanup_directory(self, directory: Path, force: bool) -> int:
        """Clean up entire directory based on policies."""
        cleaned_count = 0

        for file_path in directory.iterdir():
            if not file_path.is_file():
                continue

            try:
                if force or self._should_cleanup_file(file_path):
                    file_path.unlink()
                    cleaned_count += 1

                    # Also remove metadata file
                    meta_file = file_path.with_suffix(
                        file_path.suffix + ".meta"
                    )
                    if meta_file.exists():
                        meta_file.unlink()

            except Exception as e:
                logger.warning(f"Failed to clean up {file_path}: {e}")

        return cleaned_count

    def _should_cleanup_file(self, file_path: Path) -> bool:
        """Determine if file should be cleaned up based on policy."""
        if self.config.retention_policy == RetentionPolicy.KEEP_ALL:
            return False

        if self.config.retention_policy == RetentionPolicy.KEEP_NONE:
            return True

        if self.config.retention_policy == RetentionPolicy.KEEP_FAILURES:
            # Check if this is a failure artifact
            if self.config.preserve_failure_artifacts:
                meta_file = file_path.with_suffix(file_path.suffix + ".meta")
                if meta_file.exists():
                    try:
                        import json

                        with open(meta_file, encoding="utf-8") as f:
                            metadata = json.load(f)
                            if metadata.get("test_failed", False):
                                return False  # Keep failure artifacts
                    except Exception:
                        pass

        if self.config.retention_policy == RetentionPolicy.KEEP_RECENT:
            cutoff_time = time.time() - (
                self.config.retention_days * 24 * 3600
            )
            return file_path.stat().st_mtime < cutoff_time

        return False

    def _get_all_artifact_directories(self) -> list[Path]:
        """Get all artifact directories."""
        return [
            self.config.base_dir / self.config.screenshots_dir,
            self.config.base_dir / self.config.videos_dir,
            self.config.base_dir / self.config.regression_dir,
        ]

    def _auto_cleanup(self) -> None:
        """Perform automatic cleanup on initialization."""
        try:
            cleaned = self.cleanup_artifacts()
            if cleaned > 0:
                logger.info(f"Auto-cleanup removed {cleaned} old artifacts")
        except Exception as e:
            logger.warning(f"Auto-cleanup failed: {e}")


class CaptureStorageMixin:
    pass
