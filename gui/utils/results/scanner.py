"""Async results scanner for crack segmentation prediction triplets.

This module implements the main AsyncResultsScanner class with AsyncIO
for non-blocking I/O and semaphore concurrency control. Refactored to
comply with 400-line module limit.

Phase 1: Core AsyncIO scanner with delegated validation.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .core import ResultTriplet, ScanProgress
from .events import EventManager, ScanEvent
from .validation import (
    TripletValidator,
    group_files_by_triplet_id,
)

logger = logging.getLogger(__name__)


class AsyncResultsScanner:
    """High-performance async scanner for crack segmentation results.

    Features:
    - Non-blocking I/O using asyncio
    - Semaphore-controlled concurrency
    - Delegated triplet validation
    - Progress callbacks for UI integration
    - Error recovery and graceful degradation

    Example:
        >>> scanner = AsyncResultsScanner(results_path)
        >>> async for triplet in scanner.scan_async():
        ...     print(f"Found triplet: {triplet.id}")
    """

    def __init__(
        self,
        base_path: Path,
        max_concurrent_operations: int = 8,
        progress_callback: Callable[[ScanProgress], None] | None = None,
        event_manager: EventManager | None = None,
    ) -> None:
        """Initialize the async results scanner.

        Args:
            base_path: Root directory containing prediction results
            max_concurrent_operations: Maximum concurrent file operations
            progress_callback: Optional callback for progress updates
            event_manager: Optional event manager for reactive notifications
        """
        self.base_path = Path(base_path)
        self.max_concurrent_operations = max_concurrent_operations
        self.progress_callback = progress_callback

        # Event system for reactive updates
        self._event_manager = event_manager or EventManager()

        # Async coordination
        self._scan_lock = asyncio.Lock()
        self._semaphore: asyncio.Semaphore | None = None
        self._thread_pool: ThreadPoolExecutor | None = None

        # Validation delegate
        self._validator = TripletValidator(max_workers=4)

        # Progress tracking
        self._progress = ScanProgress()
        self._cancel_requested = False

        # Validation
        if not self.base_path.exists():
            raise FileNotFoundError(
                f"Results directory not found: {self.base_path}"
            )
        if not self.base_path.is_dir():
            raise ValueError(f"Path must be a directory: {self.base_path}")

    async def scan_async(
        self, pattern: str = "*.png", recursive: bool = True
    ) -> AsyncIterator[ResultTriplet]:
        """Scan for result triplets asynchronously.

        Args:
            pattern: File pattern to search for (default: *.png)
            recursive: Whether to scan subdirectories

        Yields:
            ResultTriplet: Complete prediction triplets found during scan
        """
        async with self._scan_lock:
            # Initialize async resources
            self._semaphore = asyncio.Semaphore(self.max_concurrent_operations)
            self._thread_pool = ThreadPoolExecutor(max_workers=4)

            try:
                # Reset progress and emit scan started event
                self._reset_progress()
                await self._event_manager.emit_async(
                    ScanEvent.scan_started(str(self.base_path), pattern)
                )

                # Discover potential triplet files
                candidate_files = await self._discover_files_async(
                    pattern, recursive
                )
                self._progress.total_files = len(candidate_files)

                # Process files and build triplets
                async for triplet in self._process_candidates_async(
                    candidate_files
                ):
                    if self._cancel_requested:
                        logger.info("Scan cancelled by user request")
                        await self._event_manager.emit_async(
                            ScanEvent.scan_cancelled(
                                "User requested cancellation"
                            )
                        )
                        break

                    # Emit triplet found event
                    await self._event_manager.emit_async(
                        ScanEvent.triplet_found(triplet)
                    )

                    yield triplet
                    self._progress.found_triplets += 1
                    self._notify_progress()

                # Emit scan completed event
                await self._event_manager.emit_async(
                    ScanEvent.scan_completed(
                        self._progress.found_triplets, self._progress.errors
                    )
                )

            except Exception as e:
                logger.error(f"Error during async scan: {e}")
                self._progress.errors += 1
                self._notify_progress()

                # Emit error event
                await self._event_manager.emit_async(
                    ScanEvent.scan_error(e, "Async scan operation")
                )
                raise
            finally:
                # Cleanup resources
                if self._thread_pool:
                    self._thread_pool.shutdown(wait=False)
                self._thread_pool = None
                self._semaphore = None
                self._validator.cleanup()

    async def _discover_files_async(
        self, pattern: str, recursive: bool
    ) -> list[Path]:
        """Discover candidate files asynchronously."""
        loop = asyncio.get_event_loop()

        def _sync_discover() -> list[Path]:
            """Synchronous file discovery for thread pool."""
            if recursive:
                return list(self.base_path.rglob(pattern))
            else:
                return list(self.base_path.glob(pattern))

        # Run in thread pool to avoid blocking event loop
        if self._thread_pool is None:
            raise RuntimeError("Thread pool not initialized")

        files = await loop.run_in_executor(self._thread_pool, _sync_discover)
        logger.info(f"Discovered {len(files)} candidate files")
        return files

    async def _process_candidates_async(
        self, candidate_files: list[Path]
    ) -> AsyncIterator[ResultTriplet]:
        """Process candidate files to build complete triplets."""
        # Group files by potential triplet ID using utility function
        triplet_groups = group_files_by_triplet_id(candidate_files)

        # Process each group concurrently using validator
        tasks = []
        for triplet_id, file_group in triplet_groups.items():
            if self._semaphore is None:
                logger.error("Semaphore not initialized")
                continue

            task = asyncio.create_task(
                self._validator.validate_triplet_async(
                    triplet_id, file_group, self._semaphore
                )
            )
            tasks.append(task)

        # Yield valid triplets as they complete
        for completed_task in asyncio.as_completed(tasks):
            try:
                triplet = await completed_task
                if triplet is not None:
                    yield triplet
            except Exception as e:
                logger.warning(f"Error processing triplet: {e}")
                self._progress.errors += 1
                self._notify_progress()
            finally:
                self._progress.processed_files += 1
                self._notify_progress()

    def _reset_progress(self) -> None:
        """Reset progress tracking."""
        self._progress = ScanProgress()
        self._cancel_requested = False

    def _notify_progress(self) -> None:
        """Notify progress callback and emit progress event."""
        # Legacy callback support
        if self.progress_callback:
            try:
                self.progress_callback(self._progress)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

        # Emit progress event (sync version for compatibility)
        try:
            self._event_manager.emit_sync(
                ScanEvent.scan_progress(self._progress)
            )
        except Exception as e:
            logger.warning(f"Progress event emission error: {e}")

    def cancel_scan(self) -> None:
        """Request cancellation of ongoing scan."""
        self._cancel_requested = True
        logger.info("Scan cancellation requested")

    @property
    def current_progress(self) -> ScanProgress:
        """Get current scan progress."""
        return self._progress


# Factory function for easy instantiation
def create_results_scanner(
    results_path: str | Path,
    max_concurrent: int = 8,
    progress_callback: Callable[[ScanProgress], None] | None = None,
    event_manager: EventManager | None = None,
) -> AsyncResultsScanner:
    """Create a configured AsyncResultsScanner instance.

    Args:
        results_path: Path to results directory
        max_concurrent: Maximum concurrent operations
        progress_callback: Optional progress callback
        event_manager: Optional event manager for reactive updates

    Returns:
        Configured AsyncResultsScanner instance
    """
    return AsyncResultsScanner(
        base_path=Path(results_path),
        max_concurrent_operations=max_concurrent,
        progress_callback=progress_callback,
        event_manager=event_manager,
    )
