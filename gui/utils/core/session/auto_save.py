"""
Auto-save system for CrackSeg GUI application.

This module provides automatic saving of user configurations to localStorage
with efficient change detection, debouncing, and visual feedback to prevent
data loss during interruptions.
"""

import json
import time
from collections.abc import Callable
from typing import Any

import streamlit as st

from crackseg.dataclasses import asdict, dataclass
from gui.utils.performance_optimizer import (
    should_update,
    track_performance,
)


@dataclass
class AutoSaveConfig:
    """Configuration for auto-save behavior."""

    save_interval: float = 3.0  # Save every 3 seconds
    debounce_interval: float = 0.5  # Wait 0.5s after last change
    max_storage_size: int = 5 * 1024 * 1024  # 5MB limit
    max_versions: int = 10  # Keep 10 versions
    enable_compression: bool = True  # Enable data compression
    visual_feedback: bool = True  # Show save indicators


@dataclass
class SaveMetadata:
    """Metadata for saved configurations."""

    timestamp: float
    version: int
    config_hash: str
    user_action: str
    size_bytes: int


class AutoSaveManager:
    """
    Comprehensive auto-save manager for GUI configurations.

    Features:
    - Automatic saving with configurable intervals
    - Change detection with debouncing
    - Visual feedback and status indicators
    - Version management with rollback support
    - Storage size management and cleanup
    """

    def __init__(self, config: AutoSaveConfig | None = None) -> None:
        """
        Initialize auto-save manager.

        Args:
            config: Auto-save configuration settings
        """
        self.config = config or AutoSaveConfig()
        self.storage_key_prefix = "crackseg_autosave"
        self.last_save_time = 0.0
        self.pending_changes: dict[str, Any] = {}
        self.change_callbacks: dict[str, Callable[[str, Any], None]] = {}
        self.save_in_progress = False

        # Performance tracking
        self.performance_id = "autosave_manager"

        # Initialize session state
        self._initialize_session_state()

    def _initialize_session_state(self) -> None:
        """Initialize Streamlit session state for auto-save."""
        if "autosave_enabled" not in st.session_state:
            st.session_state.autosave_enabled = True
        if "autosave_last_status" not in st.session_state:
            st.session_state.autosave_last_status = "idle"
        if "autosave_last_save" not in st.session_state:
            st.session_state.autosave_last_save = 0.0
        if "autosave_version" not in st.session_state:
            st.session_state.autosave_version = 1

    def register_config_field(
        self,
        field_name: str,
        value: Any,
        on_change: Callable[[str, Any], None] | None = None,
    ) -> None:
        """
        Register a configuration field for auto-save monitoring.

        Args:
            field_name: Unique identifier for the field
            value: Current value of the field
            on_change: Optional callback when field changes
        """
        start_time = time.time()

        # Check if value has changed
        current_value = self.pending_changes.get(field_name)
        if current_value != value:
            self.pending_changes[field_name] = value

            # Register change callback
            if on_change:
                self.change_callbacks[field_name] = on_change

            # Update session state timestamp
            st.session_state.autosave_last_change = time.time()

        # Track performance
        track_performance(self.performance_id, "register_field", start_time)

    def should_auto_save(self) -> bool:
        """
        Determine if auto-save should be triggered.

        Returns:
            True if auto-save should be triggered, False otherwise
        """
        if not st.session_state.autosave_enabled or self.save_in_progress:
            return False

        if not self.pending_changes:
            return False

        current_time = time.time()

        # Check if enough time has passed since last save
        time_since_save = current_time - self.last_save_time
        if time_since_save < self.config.save_interval:
            return False

        # Check debounce interval (time since last change)
        last_change = st.session_state.get("autosave_last_change", 0.0)
        time_since_change = current_time - last_change
        if time_since_change < self.config.debounce_interval:
            return False

        return True

    def auto_save_configurations(self, user_action: str = "auto") -> bool:
        """
        Perform auto-save of pending configurations.

        Args:
            user_action: Description of the user action triggering save

        Returns:
            True if save was successful, False otherwise
        """
        if not self.should_auto_save():
            return False

        start_time = time.time()
        self.save_in_progress = True

        try:
            # Update status
            st.session_state.autosave_last_status = "saving"

            # Prepare save data
            save_data = {
                "configurations": dict(self.pending_changes),
                "metadata": asdict(
                    SaveMetadata(
                        timestamp=time.time(),
                        version=st.session_state.autosave_version,
                        config_hash=self._calculate_config_hash(),
                        user_action=user_action,
                        size_bytes=0,  # Will be calculated after serialization
                    )
                ),
            }

            # Serialize data
            serialized_data = json.dumps(save_data)
            save_data["metadata"]["size_bytes"] = len(serialized_data.encode())

            # Check storage size limits
            if not self._check_storage_limits(len(serialized_data.encode())):
                st.session_state.autosave_last_status = "storage_full"
                return False

            # Save to localStorage (simulated with session_state)
            storage_key = (
                f"{self.storage_key_prefix}_v"
                f"{st.session_state.autosave_version}"
            )
            self._save_to_storage(storage_key, serialized_data)

            # Update state
            self.last_save_time = time.time()
            st.session_state.autosave_last_save = self.last_save_time
            st.session_state.autosave_last_status = "saved"
            st.session_state.autosave_version += 1

            # Clear pending changes
            self.pending_changes.clear()

            # Show visual feedback
            if self.config.visual_feedback:
                self._show_save_feedback("saved")

            # Cleanup old versions
            self._cleanup_old_versions()

            # Track performance
            track_performance(self.performance_id, "auto_save", start_time)

            return True

        except Exception as e:
            st.session_state.autosave_last_status = "error"
            if self.config.visual_feedback:
                self._show_save_feedback("error", str(e))
            return False

        finally:
            self.save_in_progress = False

    def load_saved_configuration(
        self, version: int | None = None
    ) -> dict[str, Any] | None:
        """
        Load saved configuration from storage.

        Args:
            version: Specific version to load (None for latest)

        Returns:
            Loaded configuration data or None if not found
        """
        start_time = time.time()

        try:
            if version is None:
                # Find latest version
                version = self._find_latest_version()

            if version is None:
                return None

            storage_key = f"{self.storage_key_prefix}_v{version}"
            saved_data = self._load_from_storage(storage_key)

            if saved_data:
                parsed_data = json.loads(saved_data)

                # Track performance
                track_performance(
                    self.performance_id, "load_config", start_time
                )

                return parsed_data.get("configurations", {})

            return None

        except Exception:
            return None

    def get_save_status(self) -> dict[str, Any]:
        """
        Get current auto-save status information.

        Returns:
            Dictionary with status information
        """
        return {
            "enabled": st.session_state.autosave_enabled,
            "status": st.session_state.autosave_last_status,
            "last_save": st.session_state.autosave_last_save,
            "version": st.session_state.autosave_version,
            "pending_changes": len(self.pending_changes),
            "has_saved_data": self._find_latest_version() is not None,
        }

    def toggle_auto_save(self, enabled: bool) -> None:
        """
        Enable or disable auto-save functionality.

        Args:
            enabled: Whether to enable auto-save
        """
        st.session_state.autosave_enabled = enabled
        if not enabled:
            st.session_state.autosave_last_status = "disabled"

    def force_save(self, user_action: str = "manual") -> bool:
        """
        Force immediate save regardless of intervals.

        Args:
            user_action: Description of the user action

        Returns:
            True if save was successful
        """
        # Temporarily bypass all interval checks
        self.last_save_time = 0.0
        st.session_state.autosave_last_change = 0.0
        return self.auto_save_configurations(user_action)

    def _calculate_config_hash(self) -> str:
        """Calculate hash of current configuration for change detection."""
        config_str = json.dumps(self.pending_changes, sort_keys=True)
        return str(hash(config_str))

    def _check_storage_limits(self, new_size: int) -> bool:
        """
        Check if new save would exceed storage limits.

        Args:
            new_size: Size of new data in bytes

        Returns:
            True if save is within limits
        """
        current_usage = self._calculate_storage_usage()
        return (current_usage + new_size) <= self.config.max_storage_size

    def _calculate_storage_usage(self) -> int:
        """Calculate current storage usage in bytes."""
        total_size = 0
        for version in range(1, st.session_state.autosave_version):
            key = f"{self.storage_key_prefix}_v{version}"
            if key in st.session_state:
                data = st.session_state[key]
                if isinstance(data, str):
                    total_size += len(data.encode())
        return total_size

    def _save_to_storage(self, key: str, data: str) -> None:
        """Save data to storage (localStorage simulation)."""
        st.session_state[key] = data

    def _load_from_storage(self, key: str) -> str | None:
        """Load data from storage."""
        return st.session_state.get(key)

    def _find_latest_version(self) -> int | None:
        """Find the latest saved version."""
        for version in range(st.session_state.autosave_version - 1, 0, -1):
            key = f"{self.storage_key_prefix}_v{version}"
            if key in st.session_state:
                return version
        return None

    def _cleanup_old_versions(self) -> None:
        """Clean up old versions to maintain storage limits."""
        versions_to_keep = self.config.max_versions
        current_version = st.session_state.autosave_version

        # Remove versions older than the limit
        for version in range(1, current_version - versions_to_keep):
            key = f"{self.storage_key_prefix}_v{version}"
            if key in st.session_state:
                del st.session_state[key]

    def _show_save_feedback(self, status: str, message: str = "") -> None:
        """Show visual feedback for save operations."""
        if not should_update("autosave_feedback", 1.0):
            return

        if status == "saved":
            # Use success message (will be cleared by the UI component)
            st.session_state.autosave_feedback = {
                "type": "success",
                "message": f"✅ Auto-saved at {time.strftime('%H:%M:%S')}",
                "timestamp": time.time(),
            }
        elif status == "error":
            st.session_state.autosave_feedback = {
                "type": "error",
                "message": f"❌ Auto-save failed: {message}",
                "timestamp": time.time(),
            }
        elif status == "storage_full":
            st.session_state.autosave_feedback = {
                "type": "warning",
                "message": "⚠️ Storage limit reached. Please clear old saves.",
                "timestamp": time.time(),
            }


class AutoSaveUI:
    """UI components for auto-save functionality."""

    @staticmethod
    def render_save_status(manager: AutoSaveManager) -> None:
        """
        Render auto-save status indicator.

        Args:
            manager: AutoSaveManager instance
        """
        status_info = manager.get_save_status()

        # Status indicator
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            if status_info["enabled"]:
                if status_info["status"] == "saved":
                    last_save_time = time.strftime(
                        "%H:%M:%S", time.localtime(status_info["last_save"])
                    )
                    st.success(f"🟢 Auto-save: Last saved at {last_save_time}")
                elif status_info["status"] == "saving":
                    st.info("🔄 Auto-save: Saving...")
                elif status_info["status"] == "error":
                    st.error("🔴 Auto-save: Error occurred")
                else:
                    st.info("🟡 Auto-save: Enabled")
            else:
                st.warning("⚪ Auto-save: Disabled")

        with col2:
            if st.button("💾 Save Now"):
                if manager.force_save("manual"):
                    st.success("Saved!")
                else:
                    st.error("Save failed!")

        with col3:
            current_state = status_info["enabled"]
            new_state = st.checkbox("Enable Auto-save", value=current_state)
            if new_state != current_state:
                manager.toggle_auto_save(new_state)

    @staticmethod
    def render_version_manager(manager: AutoSaveManager) -> None:
        """
        Render version management UI.

        Args:
            manager: AutoSaveManager instance
        """
        st.subheader("📚 Saved Versions")

        status_info = manager.get_save_status()

        if status_info["has_saved_data"]:
            latest_version = manager._find_latest_version()

            if latest_version:
                versions = list(
                    range(max(1, latest_version - 10), latest_version + 1)
                )

                selected_version = st.selectbox(
                    "Select version to load:",
                    options=versions,
                    index=len(versions) - 1,
                    format_func=lambda x: (
                        f"Version {x} "
                        f"{'(Latest)' if x == latest_version else ''}"
                    ),
                )

                col1, col2 = st.columns(2)

                with col1:
                    if st.button("📥 Load Selected Version"):
                        loaded_config = manager.load_saved_configuration(
                            selected_version
                        )
                        if loaded_config:
                            st.session_state.loaded_autosave_config = (
                                loaded_config
                            )
                            st.success(f"Loaded version {selected_version}")
                        else:
                            st.error("Failed to load version")

                with col2:
                    if st.button("🗑️ Clear All Saves"):
                        # Clear all auto-save data
                        for key in list(st.session_state.keys()):
                            if isinstance(key, str) and key.startswith(
                                manager.storage_key_prefix
                            ):
                                del st.session_state[key]
                        st.success("All auto-save data cleared")
        else:
            st.info("No saved versions available")

    @staticmethod
    def show_feedback_notifications() -> None:
        """Show auto-save feedback notifications."""
        if "autosave_feedback" in st.session_state:
            feedback = st.session_state.autosave_feedback

            # Show feedback for 3 seconds
            if time.time() - feedback["timestamp"] < 3.0:
                if feedback["type"] == "success":
                    st.success(feedback["message"])
                elif feedback["type"] == "error":
                    st.error(feedback["message"])
                elif feedback["type"] == "warning":
                    st.warning(feedback["message"])
            else:
                # Clear old feedback
                del st.session_state.autosave_feedback


# Global auto-save manager instance
_autosave_manager: AutoSaveManager | None = None


def get_autosave_manager(
    config: AutoSaveConfig | None = None,
) -> AutoSaveManager:
    """
    Get the global auto-save manager instance.

    Args:
        config: Optional configuration for first initialization

    Returns:
        AutoSaveManager instance
    """
    global _autosave_manager
    if _autosave_manager is None:
        _autosave_manager = AutoSaveManager(config)
    return _autosave_manager


def auto_save_field(
    field_name: str, value: Any, on_change: Callable[..., None] | None = None
) -> None:
    """
    Convenience function to register a field for auto-save.

    Args:
        field_name: Unique identifier for the field
        value: Current value of the field
        on_change: Optional callback when field changes
    """
    manager = get_autosave_manager()
    manager.register_config_field(field_name, value, on_change)


def trigger_auto_save(user_action: str = "user_input") -> bool:
    """
    Convenience function to trigger auto-save.

    Args:
        user_action: Description of the user action

    Returns:
        True if save was successful
    """
    manager = get_autosave_manager()
    return manager.auto_save_configurations(user_action)
