"""Triplet validation logic for crack segmentation results.

This module handles the validation of prediction triplets ensuring
completeness and file integrity. Separated from main scanner for
single responsibility compliance.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .core import ResultTriplet, TripletHealth, TripletType

logger = logging.getLogger(__name__)


class TripletValidator:
    """Validates crack segmentation prediction triplets.

    Handles async validation of image|mask|prediction triplets with
    file existence checks, type classification, and metadata extraction.
    """

    def __init__(self, max_workers: int = 4) -> None:
        """Initialize validator with thread pool.

        Args:
            max_workers: Maximum worker threads for I/O operations
        """
        self.max_workers = max_workers
        self._thread_pool: ThreadPoolExecutor | None = None

    async def validate_triplet_async(
        self,
        triplet_id: str,
        file_group: list[Path],
        semaphore: asyncio.Semaphore,
    ) -> ResultTriplet | None:
        """Validate and create a triplet from file group.

        Args:
            triplet_id: Unique identifier for the triplet
            file_group: List of files belonging to this triplet
            semaphore: Concurrency control semaphore

        Returns:
            Valid ResultTriplet or None if validation fails
        """
        async with semaphore:  # Control concurrency
            # Initialize thread pool if needed
            if self._thread_pool is None:
                self._thread_pool = ThreadPoolExecutor(
                    max_workers=self.max_workers
                )

            # Classify files by type
            image_path: Path | None = None
            mask_path: Path | None = None
            pred_path: Path | None = None

            for file_path in file_group:
                file_type = self._classify_file_type(file_path)

                if file_type == TripletType.IMAGE:
                    image_path = file_path
                elif file_type == TripletType.MASK:
                    mask_path = file_path
                elif file_type == TripletType.PREDICTION:
                    pred_path = file_path

            # If some paths are missing, construct expected paths for them
            # to allow for a 'DEGRADED' triplet to be created.
            if not all([image_path, mask_path, pred_path]):
                base_path = next(
                    (p for p in [image_path, mask_path, pred_path] if p),
                    None,
                )
                if base_path:
                    base_name = extract_triplet_id(base_path)
                    parent_dir = base_path.parent
                    if not image_path:
                        image_path = parent_dir / f"{base_name}.png"
                    if not mask_path:
                        mask_path = parent_dir / f"{base_name}_mask.png"
                    if not pred_path:
                        pred_path = parent_dir / f"{base_name}_pred.png"
                else:
                    # Should not happen if file_group is not empty
                    return None

            # Type narrowing - we now have paths for all, even if they don't
            # exist
            assert image_path is not None
            assert mask_path is not None
            assert pred_path is not None

            # Create triplet - its internal health check will determine its
            # status
            try:
                triplet = ResultTriplet(
                    id=triplet_id,
                    image_path=image_path,
                    mask_path=mask_path,
                    prediction_path=pred_path,
                    dataset_name=image_path.parent.name,
                    metadata=self._extract_metadata(
                        image_path, mask_path, pred_path
                    ),
                )

                if triplet.health_status == TripletHealth.BROKEN:
                    logger.debug(
                        f"Skipping broken triplet {triplet_id} "
                        "(all files missing)."
                    )
                    return None

                logger.debug(
                    f"Created triplet {triplet_id} with status "
                    f"{triplet.health_status.name}"
                )
                return triplet

            except Exception as e:
                logger.error(f"Error creating triplet {triplet_id}: {e}")
                return None

    def _classify_file_type(self, file_path: Path) -> TripletType:
        """Classify file type based on filename patterns.

        Args:
            file_path: Path to classify

        Returns:
            TripletType classification
        """
        stem_lower = file_path.stem.lower()

        if any(
            suffix in stem_lower
            for suffix in ["_mask", "_gt", "_ground_truth"]
        ):
            return TripletType.MASK
        elif any(
            suffix in stem_lower
            for suffix in ["_pred", "_prediction", "_output"]
        ):
            return TripletType.PREDICTION
        else:
            return TripletType.IMAGE

    async def _validate_files_async(self, file_paths: list[Path]) -> bool:
        """Validate file existence and basic properties asynchronously.

        Args:
            file_paths: List of paths to validate

        Returns:
            True if all files are valid
        """
        loop = asyncio.get_event_loop()

        def _sync_validate(path: Path) -> bool:
            """Synchronous file validation."""
            try:
                return (
                    path.exists()
                    and path.is_file()
                    and path.stat().st_size > 0
                )
            except (OSError, PermissionError):
                return False

        # Validate all files concurrently
        validation_tasks = [
            loop.run_in_executor(self._thread_pool, _sync_validate, path)
            for path in file_paths
        ]

        results = await asyncio.gather(
            *validation_tasks, return_exceptions=True
        )

        # Check if all validations passed
        return all(isinstance(result, bool) and result for result in results)

    def _extract_metadata(
        self, image_path: Path, mask_path: Path, pred_path: Path
    ) -> dict[str, str | int | float]:
        """Extract metadata from triplet files.

        Args:
            image_path: Path to image file
            mask_path: Path to mask file
            pred_path: Path to prediction file

        Returns:
            Dictionary with extracted metadata
        """
        try:
            return {
                "image_size": image_path.stat().st_size,
                "mask_size": mask_path.stat().st_size,
                "prediction_size": pred_path.stat().st_size,
                "parent_directory": str(image_path.parent),
                "created_time": image_path.stat().st_ctime,
            }
        except (OSError, AttributeError):
            return {}

    def cleanup(self) -> None:
        """Cleanup thread pool resources."""
        if self._thread_pool:
            self._thread_pool.shutdown(wait=False)
            self._thread_pool = None


def extract_triplet_id(file_path: Path) -> str:
    """Extract triplet ID from file path using common patterns.

    Args:
        file_path: Path to extract ID from

    Returns:
        Unique triplet identifier
    """
    stem = file_path.stem
    parent_name = file_path.parent.name

    # Remove common suffixes to get base ID
    suffixes_to_remove = [
        "_mask",
        "_pred",
        "_prediction",
        "_gt",
        "_ground_truth",
    ]

    base_stem = stem
    for suffix in suffixes_to_remove:
        if base_stem.endswith(suffix):
            base_stem = base_stem[: -len(suffix)]
            break

    # The triplet ID should be unique per directory, based on the file stem.
    # Ex: parent/abc_pred.png -> parent_abc
    return f"{parent_name}_{base_stem}"


def group_files_by_triplet_id(files: list[Path]) -> dict[str, list[Path]]:
    """Group files by their potential triplet ID.

    Args:
        files: List of files to group

    Returns:
        Dictionary mapping triplet IDs to file groups
    """
    groups: dict[str, list[Path]] = {}

    for file_path in files:
        # Extract triplet ID from filename
        triplet_id = extract_triplet_id(file_path)

        if triplet_id not in groups:
            groups[triplet_id] = []
        groups[triplet_id].append(file_path)

    return groups
