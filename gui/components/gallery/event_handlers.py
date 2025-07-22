"""
Event handlers for the Results Gallery component.

This module centralizes all event handling logic, decoupling the main
component from the details of state updates in response to scanner events.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import streamlit as st

from gui.utils.results import EventType, get_event_manager

if TYPE_CHECKING:
    import gui.components.gallery.state_manager as sm
    from gui.utils.results import ScanEvent


class GalleryEventHandlers:
    """Registers and defines event handlers for the gallery."""

    def __init__(self, state_manager: sm.GalleryStateManager) -> None:
        """Initialize the event handlers.

        Args:
            state_manager: The state manager instance for updating state.
        """
        self.state = state_manager
        self.event_manager = get_event_manager()

    def setup_event_handlers(self) -> None:
        """Register all event handlers with the event manager."""
        self.event_manager.subscribe(
            EventType.SCAN_PROGRESS, self.on_scan_progress
        )
        self.event_manager.subscribe(
            EventType.TRIPLET_FOUND, self.on_triplet_found
        )
        self.event_manager.subscribe(
            EventType.SCAN_COMPLETED, self.on_scan_completed
        )
        self.event_manager.subscribe(
            EventType.SCAN_ERROR, self.on_validation_error
        )

    def on_scan_progress(self, event: ScanEvent) -> None:
        """Handle scan progress events."""
        progress_data = event.data.get("progress")
        if progress_data:
            progress_update = {
                "current": progress_data.scanned_files,
                "total": progress_data.total_files,
                "message": f"Found {progress_data.found_triplets} triplets",
                "percent": progress_data.progress_percent,
            }
            self.state.set("scan_progress", progress_update)
            st.rerun()

    def on_triplet_found(self, event: ScanEvent) -> None:
        """Handle triplet found events."""
        triplet = event.data.get("triplet")
        if triplet:
            current_results = self.state.get("scan_results", [])
            current_results.append(triplet)
            self.state.set("scan_results", current_results)

            # Assuming cache is handled by the scanner service
            if self.state.get("config", {}).get("enable_real_time", True):
                st.rerun()

    def on_scan_completed(self, event: ScanEvent) -> None:
        """Handle scan completion events."""
        self.state.set("scan_active", False)
        self.state.set("last_scan_time", time.time())

        total_triplets = event.data.get("total_triplets", 0)
        total_errors = event.data.get("total_errors", 0)
        total_scanned = total_triplets + total_errors

        self.state.update(
            "validation_stats",
            {
                "total_scanned": total_scanned,
                "valid_triplets": total_triplets,
                "errors": total_errors,
                "success_rate": (
                    (total_triplets / total_scanned * 100.0)
                    if total_scanned > 0
                    else 0.0
                ),
                "completion_time": time.time(),
            },
        )
        st.rerun()

    def on_validation_error(self, event: ScanEvent) -> None:
        """Handle validation error events."""
        error_log = self.state.get("error_log", {})
        error_data = event.data
        error_log[error_data.get("path", "unknown")] = error_data
        self.state.set("error_log", error_log)
