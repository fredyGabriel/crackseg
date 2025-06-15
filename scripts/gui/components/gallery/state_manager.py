"""
Manages the session state for the Results Gallery component.

Encapsulates all Streamlit session state keys and provides a centralized
interface for state initialization and management, preventing key
collisions and improving code clarity.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from scripts.gui.utils.export_manager import ExportManager


class GalleryStateManager:
    """Handles session state for the Results Gallery."""

    def __init__(self, key_prefix: str = "gallery") -> None:
        """Initialize the StateManager with a unique prefix."""
        self.key_prefix = key_prefix
        self._state_keys = {
            "scan_active": f"{self.key_prefix}_scan_active",
            "scan_progress": f"{self.key_prefix}_scan_progress",
            "scan_results": f"{self.key_prefix}_scan_results",
            "selected_triplet_ids": f"{self.key_prefix}_selected_triplet_ids",
            "validation_stats": f"{self.key_prefix}_validation_stats",
            "error_log": f"{self.key_prefix}_error_log",
            "last_scan_time": f"{self.key_prefix}_last_scan_time",
            "filter_criteria": f"{self.key_prefix}_filter_criteria",
            "active_filters": f"{self.key_prefix}_active_filters",
            "export_manager": f"{self.key_prefix}_export_manager",
            "export_future": f"{self.key_prefix}_export_future",
            "export_progress": f"{self.key_prefix}_export_progress",
            "config": f"{self.key_prefix}_config",
        }
        self.initialize_state()

    def initialize_state(self) -> None:
        """Initialize all required session state variables
        if they don't exist."""
        for key, state_key in self._state_keys.items():
            if state_key not in st.session_state:
                if "progress" in key:
                    st.session_state[state_key] = {
                        "current": 0,
                        "total": 0,
                        "message": "",
                        "percent": 0.0,
                    }
                elif "results" in key:
                    st.session_state[state_key] = []
                elif "selected" in key:
                    st.session_state[state_key] = set()
                elif "stats" in key or "log" in key or "filter" in key:
                    st.session_state[state_key] = {}
                elif "config" in key:
                    st.session_state[state_key] = {}
                else:
                    st.session_state[state_key] = False

        # Special initialization for complex objects
        if self.get_key("export_manager") not in st.session_state:
            st.session_state[self.get_key("export_manager")] = ExportManager(
                on_progress=self._update_export_progress
            )

    def get_key(self, key_name: str) -> str:
        """Get the full session state key for a given short name."""
        return self._state_keys[key_name]

    def get(self, key_name: str, default: Any = None) -> Any:
        """Get a value from session state."""
        return st.session_state.get(self.get_key(key_name), default)

    def set(self, key_name: str, value: Any) -> None:
        """Set a value in session state."""
        st.session_state[self.get_key(key_name)] = value

    def update(self, key_name: str, new_values: dict[str, Any]) -> None:
        """Update a dictionary-based value in session state."""
        current_value = self.get(key_name, {})
        current_value.update(new_values)
        self.set(key_name, current_value)

    def _update_export_progress(self, percent: float, message: str) -> None:
        """Update the export progress in the session state."""
        self.set(
            "export_progress",
            {"percent": percent, "message": message},
        )
        st.rerun()

    def clear_results(self) -> None:
        """Clear all scan-related results and reset the gallery state."""
        self.set("scan_results", [])
        self.set("selected_triplet_ids", set())
        self.set("validation_stats", {})
        self.set("error_log", {})
        self.set("last_scan_time", None)
        # Assuming cache is handled elsewhere or passed in
        st.toast("Gallery cleared!", icon="🗑️")
        st.rerun()

    def reset_export_state(self) -> None:
        """Reset the export future and progress."""
        self.set("export_future", None)
        self.set(
            "export_progress",
            {"percent": 0.0, "message": ""},
        )
