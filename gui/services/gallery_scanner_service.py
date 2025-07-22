"""
Asynchronous scanning service for the Results Gallery.

This service encapsulates the logic for running scans in a background
thread, handling asynchronous operations, and emitting events through
the event manager.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

from gui.utils.results import (
    ScanEvent,
    ValidationLevel,
    create_results_scanner,
    get_event_manager,
    get_triplet_cache,
)

if TYPE_CHECKING:
    from gui.utils.results import AsyncResultsScanner, EventManager

logger = logging.getLogger(__name__)


class GalleryScannerService:
    """Manages the asynchronous scanning process."""

    def __init__(self, event_manager: EventManager | None = None) -> None:
        """Initialize the scanner service."""
        self.event_manager = event_manager or get_event_manager()
        self.cache = get_triplet_cache()
        self.scanner: AsyncResultsScanner | None = None

    def start_scan(
        self, scan_directory: str | Path, validation_level: ValidationLevel
    ) -> None:
        """Start the scan in a background thread."""
        self.scanner = create_results_scanner(
            results_path=scan_directory,
            event_manager=self.event_manager,
            # validation_level is not a direct parameter of the scanner,
            # but would be handled by a validator if separated.
        )
        self.cache.clear()

        executor = ThreadPoolExecutor(max_workers=1)
        executor.submit(self._run_scan_in_thread)

    def _run_scan_in_thread(self) -> None:
        """Run the entire async scan process in a managed thread."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._perform_scan())
        except Exception as e:
            logger.error(f"Error during scan thread: {e}")
            if self.event_manager:
                error_event = ScanEvent.scan_error(e, context="Scan Thread")
                asyncio.run(self.event_manager.emit_async(error_event))

    async def _perform_scan(self) -> None:
        """The core async scanning and validation logic."""
        if not self.scanner:
            return

        # The scanner now handles its own event emissions.
        # We just need to iterate through it to drive the process.
        async for _ in self.scanner.scan_async():
            pass  # The work is done inside the scanner and event handlers
