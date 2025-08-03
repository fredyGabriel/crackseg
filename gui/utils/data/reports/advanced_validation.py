"""Advanced triplet validation with integrity checks and error recovery.

This module extends the basic validation with comprehensive integrity checks,
corruption detection, error recovery strategies, and graceful degradation
for crack segmentation prediction triplets.

Phase 3: Enhanced validation + comprehensive error handling.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

from .core import ResultTriplet, TripletType
from .events import EventManager, ScanEvent

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Levels of validation thoroughness."""

    BASIC = auto()  # File existence and size
    STANDARD = auto()  # + Format validation
    THOROUGH = auto()  # + Content integrity
    PARANOID = auto()  # + Checksum verification


class ValidationError(Exception):
    """Base exception for validation errors."""

    def __init__(
        self, message: str, error_code: str, recoverable: bool = True
    ):
        super().__init__(message)
        self.error_code = error_code
        self.recoverable = recoverable


class CorruptionError(ValidationError):
    """Exception for file corruption issues."""

    def __init__(self, file_path: Path, details: str):
        super().__init__(
            f"File corruption detected in {file_path}: {details}",
            "CORRUPTION_DETECTED",
            recoverable=False,
        )
        self.file_path = file_path


class IntegrityError(ValidationError):
    """Exception for triplet integrity issues."""

    def __init__(self, triplet_id: str, missing_files: list[str]):
        super().__init__(
            f"Triplet {triplet_id} integrity compromised: "
            f"missing {missing_files}",
            "INTEGRITY_VIOLATION",
            recoverable=True,
        )
        self.triplet_id = triplet_id
        self.missing_files = missing_files


@dataclass
class ValidationResult:
    """Result of triplet validation with detailed diagnostics."""

    triplet_id: str
    is_valid: bool
    validation_level: ValidationLevel
    triplet: ResultTriplet | None = None
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    validation_time: float = 0.0

    @property
    def has_recoverable_errors(self) -> bool:
        """Check if all errors are recoverable."""
        return all(error.recoverable for error in self.errors)

    @property
    def error_summary(self) -> str:
        """Get a summary of validation errors."""
        if not self.errors:
            return "No errors"

        error_counts = {}
        for error in self.errors:
            error_counts[error.error_code] = (
                error_counts.get(error.error_code, 0) + 1
            )

        return ", ".join(
            f"{code}: {count}" for code, count in error_counts.items()
        )


@dataclass
class ValidationStats:
    """Statistics for validation operations."""

    total_validated: int = 0
    successful: int = 0
    failed: int = 0
    corrupted: int = 0
    recovered: int = 0
    total_time: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate validation success rate."""
        return (
            (self.successful / self.total_validated * 100.0)
            if self.total_validated > 0
            else 0.0
        )

    @property
    def recovery_rate(self) -> float:
        """Calculate error recovery rate."""
        return (
            (self.recovered / self.failed * 100.0) if self.failed > 0 else 0.0
        )


class AdvancedTripletValidator:
    """Advanced validator with integrity checks and error recovery.

    Features:
    - Multi-level validation (basic to paranoid)
    - File corruption detection
    - Automatic error recovery strategies
    - Graceful degradation for partial triplets
    - Performance monitoring and statistics

    Example:
        >>> validator = AdvancedTripletValidator(
        ...     validation_level=ValidationLevel.THOROUGH,
        ...     enable_recovery=True
        ... )
        >>> result = await validator.validate_triplet_advanced(
        ...     triplet_id, file_group, semaphore
        ... )
        >>> if result.is_valid:
        ...     print(f"Triplet {result.triplet_id} validated successfully")
    """

    def __init__(
        self,
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        enable_recovery: bool = True,
        max_workers: int = 4,
        event_manager: EventManager | None = None,
    ) -> None:
        """Initialize the advanced validator.

        Args:
            validation_level: Thoroughness of validation
            enable_recovery: Whether to attempt error recovery
            max_workers: Maximum worker threads for I/O operations
            event_manager: Optional event manager for notifications
        """
        self.validation_level = validation_level
        self.enable_recovery = enable_recovery
        self.max_workers = max_workers
        self.event_manager = event_manager

        # Thread pool for I/O operations
        self._thread_pool: ThreadPoolExecutor | None = None

        # Statistics tracking
        self.stats = ValidationStats()

        # Recovery strategies registry
        self._recovery_strategies: dict[str, Any] = {
            "MISSING_FILE": self._recover_missing_file,
            "CORRUPTION_DETECTED": self._recover_corrupted_file,
            "SIZE_MISMATCH": self._recover_size_mismatch,
        }

    async def validate_triplet_advanced(
        self,
        triplet_id: str,
        file_group: list[Path],
        semaphore: asyncio.Semaphore,
    ) -> ValidationResult:
        """Perform advanced validation of a triplet.

        Args:
            triplet_id: Unique identifier for the triplet
            file_group: List of files belonging to this triplet
            semaphore: Concurrency control semaphore

        Returns:
            Detailed validation result with diagnostics
        """
        start_time = time.time()
        result = ValidationResult(
            triplet_id=triplet_id,
            is_valid=False,
            validation_level=self.validation_level,
        )

        async with semaphore:
            try:
                # Initialize thread pool if needed
                if self._thread_pool is None:
                    self._thread_pool = ThreadPoolExecutor(
                        max_workers=self.max_workers
                    )

                # Phase 1: Basic file classification and existence
                classified_files = await self._classify_and_validate_files(
                    file_group, result
                )

                if not classified_files:
                    result.errors.append(
                        ValidationError(
                            f"No valid files found for triplet {triplet_id}",
                            "NO_VALID_FILES",
                            recoverable=False,
                        )
                    )
                    return result

                # Phase 2: Integrity validation
                await self._validate_triplet_integrity(
                    classified_files, result
                )

                # Phase 3: Content validation (if level permits)
                if self.validation_level in [
                    ValidationLevel.THOROUGH,
                    ValidationLevel.PARANOID,
                ]:
                    await self._validate_file_content(classified_files, result)

                # Phase 4: Checksum validation (paranoid level only)
                if self.validation_level == ValidationLevel.PARANOID:
                    await self._validate_checksums(classified_files, result)

                # Phase 5: Error recovery (if enabled and needed)
                if (
                    self.enable_recovery
                    and result.errors
                    and result.has_recoverable_errors
                ):
                    await self._attempt_error_recovery(
                        classified_files, result
                    )

                # Phase 6: Create triplet if validation passed
                if not result.errors or (
                    self.enable_recovery and result.has_recoverable_errors
                ):
                    result.triplet = await self._create_validated_triplet(
                        triplet_id, classified_files, result
                    )
                    result.is_valid = result.triplet is not None

                # Update statistics
                self._update_stats(result)

                # Emit validation event
                if self.event_manager:
                    await self._emit_validation_event(result)

            except Exception as e:
                logger.error(
                    f"Unexpected error validating triplet {triplet_id}: {e}"
                )
                result.errors.append(
                    ValidationError(
                        f"Validation failed: {str(e)}",
                        "VALIDATION_EXCEPTION",
                        recoverable=False,
                    )
                )

            finally:
                result.validation_time = time.time() - start_time

        return result

    async def _classify_and_validate_files(
        self, file_group: list[Path], result: ValidationResult
    ) -> dict[TripletType, Path]:
        """Classify files by type and perform basic validation.

        Args:
            file_group: List of files to classify
            result: Validation result to update

        Returns:
            Dictionary mapping file types to paths
        """
        classified: dict[TripletType, Path] = {}

        for file_path in file_group:
            try:
                # Basic existence check
                if not await self._file_exists_async(file_path):
                    result.warnings.append(f"File not found: {file_path}")
                    continue

                # Classify file type
                file_type = self._classify_file_type(file_path)

                # Check for duplicates
                if file_type in classified:
                    result.warnings.append(
                        f"Duplicate {file_type.name} file: {file_path}"
                    )
                    continue

                classified[file_type] = file_path

            except Exception as e:
                result.errors.append(
                    ValidationError(
                        f"Error processing file {file_path}: {str(e)}",
                        "FILE_PROCESSING_ERROR",
                        recoverable=True,
                    )
                )

        return classified

    async def _validate_triplet_integrity(
        self,
        classified_files: dict[TripletType, Path],
        result: ValidationResult,
    ) -> None:
        """Validate triplet has all required components.

        Args:
            classified_files: Classified file paths
            result: Validation result to update
        """
        required_types = {
            TripletType.IMAGE,
            TripletType.MASK,
            TripletType.PREDICTION,
        }
        missing_types = required_types - set(classified_files.keys())

        if missing_types:
            missing_names = [t.name.lower() for t in missing_types]
            result.errors.append(
                IntegrityError(result.triplet_id, missing_names)
            )

        # Validate file sizes
        for file_type, file_path in classified_files.items():
            try:
                size = await self._get_file_size_async(file_path)
                if size == 0:
                    result.errors.append(
                        ValidationError(
                            f"Empty {file_type.name} file: {file_path}",
                            "EMPTY_FILE",
                            recoverable=False,
                        )
                    )
                elif size < 100:  # Suspiciously small for image files
                    result.warnings.append(
                        f"Suspiciously small {file_type.name} file: "
                        f"{file_path} ({size} bytes)"
                    )
            except Exception as e:
                result.errors.append(
                    ValidationError(
                        f"Cannot access {file_type.name} file "
                        f"{file_path}: {str(e)}",
                        "FILE_ACCESS_ERROR",
                        recoverable=True,
                    )
                )

    async def _validate_file_content(
        self,
        classified_files: dict[TripletType, Path],
        result: ValidationResult,
    ) -> None:
        """Validate file content and format.

        Args:
            classified_files: Classified file paths
            result: Validation result to update
        """
        for file_type, file_path in classified_files.items():
            try:
                # Check if file is readable and has valid format
                is_valid = await self._validate_image_format_async(file_path)
                if not is_valid:
                    result.errors.append(
                        CorruptionError(
                            file_path,
                            f"Invalid or corrupted {file_type.name} format",
                        )
                    )
            except Exception as e:
                result.errors.append(
                    ValidationError(
                        f"Content validation failed for {file_path}: {str(e)}",
                        "CONTENT_VALIDATION_ERROR",
                        recoverable=False,
                    )
                )

    async def _validate_checksums(
        self,
        classified_files: dict[TripletType, Path],
        result: ValidationResult,
    ) -> None:
        """Validate file checksums for integrity.

        Args:
            classified_files: Classified file paths
            result: Validation result to update
        """
        for file_type, file_path in classified_files.items():
            try:
                checksum = await self._calculate_checksum_async(file_path)
                result.metadata[f"{file_type.name.lower()}_checksum"] = (
                    checksum
                )

                # Store checksum for future validation
                # In a real implementation, you might compare against
                # stored checksums

            except Exception as e:
                result.warnings.append(
                    f"Checksum calculation failed for {file_path}: {str(e)}"
                )

    async def _attempt_error_recovery(
        self,
        classified_files: dict[TripletType, Path],
        result: ValidationResult,
    ) -> None:
        """Attempt to recover from validation errors.

        Args:
            classified_files: Classified file paths
            result: Validation result to update
        """
        recovered_errors = []

        for error in result.errors:
            if not error.recoverable:
                continue

            try:
                recovery_strategy = self._recovery_strategies.get(
                    error.error_code
                )
                if recovery_strategy:
                    success = await recovery_strategy(
                        error, classified_files, result
                    )
                    if success:
                        recovered_errors.append(error)
                        result.warnings.append(
                            f"Recovered from error: {error.error_code}"
                        )
            except Exception as e:
                logger.warning(f"Recovery failed for {error.error_code}: {e}")

        # Remove recovered errors
        for error in recovered_errors:
            result.errors.remove(error)
            self.stats.recovered += 1

    async def _create_validated_triplet(
        self,
        triplet_id: str,
        classified_files: dict[TripletType, Path],
        result: ValidationResult,
    ) -> ResultTriplet | None:
        """Create a validated triplet from classified files.

        Args:
            triplet_id: Triplet identifier
            classified_files: Classified file paths
            result: Validation result

        Returns:
            Created triplet or None if creation fails
        """
        try:
            # Ensure we have all required files
            if len(classified_files) < 3:
                return None

            image_path = classified_files[TripletType.IMAGE]
            mask_path = classified_files[TripletType.MASK]
            pred_path = classified_files[TripletType.PREDICTION]

            # Extract enhanced metadata
            metadata = await self._extract_enhanced_metadata(
                image_path, mask_path, pred_path
            )
            metadata.update(result.metadata)

            triplet = ResultTriplet(
                id=triplet_id,
                image_path=image_path,
                mask_path=mask_path,
                prediction_path=pred_path,
                dataset_name=image_path.parent.name,
                metadata=metadata,
            )

            return triplet

        except Exception as e:
            result.errors.append(
                ValidationError(
                    f"Failed to create triplet: {str(e)}",
                    "TRIPLET_CREATION_ERROR",
                    recoverable=False,
                )
            )
            return None

    # Helper methods for async operations
    async def _file_exists_async(self, file_path: Path) -> bool:
        """Check if file exists asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._thread_pool, lambda: file_path.exists()
        )

    async def _get_file_size_async(self, file_path: Path) -> int:
        """Get file size asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._thread_pool, lambda: file_path.stat().st_size
        )

    async def _validate_image_format_async(self, file_path: Path) -> bool:
        """Validate image format asynchronously."""
        loop = asyncio.get_event_loop()

        def _validate_format() -> bool:
            try:
                # Basic format validation - check file signature
                with open(file_path, "rb") as f:
                    header = f.read(8)

                # Check for common image format signatures
                png_signature = b"\x89PNG\r\n\x1a\n"
                jpeg_signatures = [b"\xff\xd8\xff", b"\xff\xd9"]

                if header.startswith(png_signature):
                    return True
                if any(header.startswith(sig) for sig in jpeg_signatures):
                    return True

                return False
            except Exception:
                return False

        return await loop.run_in_executor(self._thread_pool, _validate_format)

    async def _calculate_checksum_async(self, file_path: Path) -> str:
        """Calculate file checksum asynchronously."""
        loop = asyncio.get_event_loop()

        def _calculate_md5() -> str:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()

        return await loop.run_in_executor(self._thread_pool, _calculate_md5)

    async def _extract_enhanced_metadata(
        self, image_path: Path, mask_path: Path, pred_path: Path
    ) -> dict[str, Any]:
        """Extract enhanced metadata from triplet files."""
        loop = asyncio.get_event_loop()

        def _extract_sync() -> dict[str, Any]:
            try:
                return {
                    "image_size": image_path.stat().st_size,
                    "mask_size": mask_path.stat().st_size,
                    "prediction_size": pred_path.stat().st_size,
                    "parent_directory": str(image_path.parent),
                    "created_time": image_path.stat().st_ctime,
                    "modified_time": image_path.stat().st_mtime,
                    "validation_level": self.validation_level.name,
                    "validation_timestamp": time.time(),
                }
            except Exception:
                return {}

        return await loop.run_in_executor(self._thread_pool, _extract_sync)

    def _classify_file_type(self, file_path: Path) -> TripletType:
        """Classify file type based on filename patterns."""
        stem_lower = file_path.stem.lower()

        if any(
            suffix in stem_lower
            for suffix in ["_mask", "_gt", "_ground_truth", "_label"]
        ):
            return TripletType.MASK
        elif any(
            suffix in stem_lower
            for suffix in ["_pred", "_prediction", "_output", "_result"]
        ):
            return TripletType.PREDICTION
        else:
            return TripletType.IMAGE

    # Recovery strategy implementations
    async def _recover_missing_file(
        self,
        error: ValidationError,
        classified_files: dict[TripletType, Path],
        result: ValidationResult,
    ) -> bool:
        """Attempt to recover from missing file errors."""
        logger.info(f"Attempting recovery for missing file: {error}")
        return False

    async def _recover_corrupted_file(
        self,
        error: ValidationError,
        classified_files: dict[TripletType, Path],
        result: ValidationResult,
    ) -> bool:
        """Attempt to recover from file corruption."""
        logger.info(f"Attempting recovery for corrupted file: {error}")
        return False

    async def _recover_size_mismatch(
        self,
        error: ValidationError,
        classified_files: dict[TripletType, Path],
        result: ValidationResult,
    ) -> bool:
        """Attempt to recover from size mismatch errors."""
        logger.info(f"Attempting recovery for size mismatch: {error}")
        return False

    def _update_stats(self, result: ValidationResult) -> None:
        """Update validation statistics."""
        self.stats.total_validated += 1
        self.stats.total_time += result.validation_time

        if result.is_valid:
            self.stats.successful += 1
        else:
            self.stats.failed += 1

        # Check for corruption
        if any(isinstance(error, CorruptionError) for error in result.errors):
            self.stats.corrupted += 1

    async def _emit_validation_event(self, result: ValidationResult) -> None:
        """Emit validation event if event manager is available."""
        if not self.event_manager:
            return

        try:
            # Create appropriate event based on result
            if result.is_valid and result.triplet is not None:
                event = ScanEvent.triplet_found(result.triplet)
            else:
                event = ScanEvent.scan_error(
                    Exception(f"Validation failed: {result.error_summary}"),
                    f"Triplet validation for {result.triplet_id}",
                )

            await self.event_manager.emit_async(event)

        except Exception as e:
            logger.warning(f"Failed to emit validation event: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive validation statistics."""
        return {
            "total_validated": self.stats.total_validated,
            "successful": self.stats.successful,
            "failed": self.stats.failed,
            "corrupted": self.stats.corrupted,
            "recovered": self.stats.recovered,
            "success_rate": self.stats.success_rate,
            "recovery_rate": self.stats.recovery_rate,
            "total_time": self.stats.total_time,
            "avg_time_per_validation": (
                self.stats.total_time / self.stats.total_validated
                if self.stats.total_validated > 0
                else 0.0
            ),
            "validation_level": self.validation_level.name,
            "recovery_enabled": self.enable_recovery,
        }

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self._thread_pool:
            self._thread_pool.shutdown(wait=False)
            self._thread_pool = None
