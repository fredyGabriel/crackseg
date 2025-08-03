"""
User action handler for the Results Gallery component.

This module centralizes the logic that responds to user interactions
(e.g., button clicks), acting as a bridge between the UI renderer
and the backend services (scanner, exporter) and state manager.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import gui.components.gallery.state_manager as sm
    from gui.services.gallery_export_service import (
        GalleryExportService,
    )
    from gui.services.gallery_scanner_service import (
        GalleryScannerService,
    )
    from gui.utils.results import ValidationLevel


class GalleryActions:
    """Handles user-triggered actions from the gallery UI."""

    def __init__(
        self,
        state: sm.GalleryStateManager,
        scanner_service: GalleryScannerService,
        exporter_service: GalleryExportService,
    ) -> None:
        """Initialize the actions handler.

        Args:
            state: The state manager instance.
            scanner_service: The gallery scanner service instance.
            exporter_service: The gallery exporter service instance.
        """
        self.state = state
        self.scanner = scanner_service
        self.exporter = exporter_service

    def start_scan(
        self, scan_directory: str | Path, validation_level: ValidationLevel
    ) -> None:
        """Action to start a new scan."""
        self.state.set("scan_active", True)
        self.state.set("scan_results", [])
        self.state.set("error_log", {})
        self.scanner.start_scan(scan_directory, validation_level)

    def clear_results(self) -> None:
        """Action to clear all results and reset the gallery."""
        self.state.clear_results()
        # self.cache.clear() # Cache should be handled by the service

    def handle_export(
        self,
        format_type: str,
        scope: str,
        include_images: bool,
        include_metadata: bool,
    ) -> None:
        """Action to start an export job."""
        self.exporter.handle_export(scope, include_images, include_metadata)

    def invert_selection(self) -> None:
        """Invert the current selection of triplets."""
        # This logic will need access to filtered results
        pass

    def select_all_filtered(self) -> None:
        """Select all triplets that are currently visible after filtering."""
        pass

    def clear_all_selections(self) -> None:
        """Clear all triplet selections."""
        self.state.set("selected_triplet_ids", set())
