"""
Auto-save manager component for CrackSeg GUI.

This component provides a Streamlit interface for the auto-save functionality,
integrating seamlessly with the main GUI application.
"""

from typing import Any

import streamlit as st

from scripts.gui.components.loading_spinner_optimized import (
    OptimizedLoadingSpinner,
)
from scripts.gui.utils.auto_save import (
    AutoSaveConfig,
    AutoSaveManager,
    AutoSaveUI,
    get_autosave_manager,
)


class AutoSaveManagerComponent:
    """
    Streamlit component for auto-save management.

    This component provides a complete UI for managing auto-save functionality
    within the CrackSeg GUI application.
    """

    def __init__(self, config: AutoSaveConfig | None = None) -> None:
        """
        Initialize the auto-save manager component.

        Args:
            config: Optional auto-save configuration
        """
        # If config is provided, create a new manager instance
        # Otherwise, use the global singleton
        if config is not None:
            self.manager = AutoSaveManager(config)
        else:
            self.manager = get_autosave_manager()
        self.spinner = OptimizedLoadingSpinner()

    def render_auto_save_panel(self) -> None:
        """Render the main auto-save control panel."""
        st.subheader("💾 Auto-Save Manager")

        # Show feedback notifications first
        AutoSaveUI.show_feedback_notifications()

        # Main status and controls
        AutoSaveUI.render_save_status(self.manager)

        # Additional controls
        self._render_advanced_controls()

        # Version management
        if st.expander("📚 Version Management", expanded=False):
            AutoSaveUI.render_version_manager(self.manager)

    def render_compact_status(self) -> None:
        """Render a compact status indicator for the main interface."""
        status_info = self.manager.get_save_status()

        # Compact status in sidebar or header
        col1, col2 = st.columns([3, 1])

        with col1:
            if status_info["enabled"]:
                if status_info["status"] == "saved":
                    st.success("✅ Auto-saved", icon="✅")
                elif status_info["status"] == "saving":
                    st.info("🔄 Saving...", icon="🔄")
                elif status_info["status"] == "error":
                    st.error("❌ Save error", icon="❌")
                else:
                    st.info("💾 Auto-save ON", icon="💾")
            else:
                st.warning("⚪ Auto-save OFF", icon="⚪")

        with col2:
            if st.button("💾", help="Save now"):
                if self.manager.force_save("manual"):
                    st.success("Saved!")
                else:
                    st.error("Failed!")

    def integrate_with_config_form(self, config_data: dict[str, Any]) -> None:
        """
        Integrate auto-save with configuration forms.

        Args:
            config_data: Configuration data to monitor for changes
        """
        # Register all config fields for auto-save monitoring
        for field_name, value in config_data.items():
            self.manager.register_config_field(field_name, value)

        # Trigger auto-save check
        if self.manager.should_auto_save():
            self.manager.auto_save_configurations("form_change")

    def load_configuration_if_available(self) -> dict[str, Any] | None:
        """
        Load saved configuration if available.

        Returns:
            Loaded configuration or None if not available
        """
        if "loaded_autosave_config" in st.session_state:
            config = st.session_state.loaded_autosave_config
            del st.session_state.loaded_autosave_config
            return config

        return self.manager.load_saved_configuration()

    def _render_advanced_controls(self) -> None:
        """Render advanced auto-save controls."""
        with st.expander("⚙️ Advanced Settings", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Timing")
                current_config = self.manager.config

                # Save interval
                save_interval = st.slider(
                    "Save interval (seconds)",
                    min_value=1.0,
                    max_value=30.0,
                    value=current_config.save_interval,
                    step=0.5,
                    help="How often to auto-save (minimum time between saves)",
                )

                # Debounce interval
                debounce_interval = st.slider(
                    "Debounce interval (seconds)",
                    min_value=0.1,
                    max_value=5.0,
                    value=current_config.debounce_interval,
                    step=0.1,
                    help="Wait time after last change before saving",
                )

                # Update config if changed
                if (
                    save_interval != current_config.save_interval
                    or debounce_interval != current_config.debounce_interval
                ):
                    self.manager.config.save_interval = save_interval
                    self.manager.config.debounce_interval = debounce_interval

            with col2:
                st.subheader("Storage")

                # Max versions
                max_versions = st.slider(
                    "Max versions to keep",
                    min_value=1,
                    max_value=50,
                    value=current_config.max_versions,
                    help="Maximum number of saved versions to keep",
                )

                # Visual feedback
                visual_feedback = st.checkbox(
                    "Show visual feedback",
                    value=current_config.visual_feedback,
                    help="Show save notifications and status messages",
                )

                # Update config if changed
                if (
                    max_versions != current_config.max_versions
                    or visual_feedback != current_config.visual_feedback
                ):
                    self.manager.config.max_versions = max_versions
                    self.manager.config.visual_feedback = visual_feedback

                # Storage info
                status_info = self.manager.get_save_status()
                st.metric("Current version", status_info["version"])
                st.metric("Pending changes", status_info["pending_changes"])

    def render_debug_info(self) -> None:
        """Render debug information for development."""
        if st.checkbox("Show debug info", value=False):
            st.subheader("🔍 Debug Information")

            status_info = self.manager.get_save_status()

            col1, col2 = st.columns(2)

            with col1:
                st.json(
                    {
                        "Status": status_info["status"],
                        "Enabled": status_info["enabled"],
                        "Version": status_info["version"],
                        "Pending Changes": status_info["pending_changes"],
                        "Has Saved Data": status_info["has_saved_data"],
                    }
                )

            with col2:
                st.json(
                    {
                        "Save Interval": self.manager.config.save_interval,
                        "Debounce Interval": (
                            self.manager.config.debounce_interval
                        ),
                        "Max Versions": self.manager.config.max_versions,
                        "Visual Feedback": self.manager.config.visual_feedback,
                        "Last Save Time": self.manager.last_save_time,
                        "Save In Progress": self.manager.save_in_progress,
                    }
                )

            # Manual operations
            st.subheader("Manual Operations")

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("🔄 Force Save"):
                    result = self.manager.force_save("debug_manual")
                    if result:
                        st.success("Force save successful")
                    else:
                        st.error("Force save failed")

            with col2:
                if st.button("📥 Load Latest"):
                    config = self.manager.load_saved_configuration()
                    if config:
                        st.success("Loaded configuration")
                        st.json(config)
                    else:
                        st.error("No configuration found")

            with col3:
                if st.button("🗑️ Clear All"):
                    # Clear all auto-save data
                    for key in list(st.session_state.keys()):
                        if isinstance(key, str) and key.startswith(
                            self.manager.storage_key_prefix
                        ):
                            del st.session_state[key]
                    st.success("All auto-save data cleared")


def create_auto_save_manager(
    config: AutoSaveConfig | None = None,
) -> AutoSaveManagerComponent:
    """
    Factory function to create an auto-save manager component.

    Args:
        config: Optional auto-save configuration

    Returns:
        AutoSaveManagerComponent instance
    """
    return AutoSaveManagerComponent(config)


def integrate_auto_save_with_page(
    page_name: str, config_data: dict[str, Any]
) -> AutoSaveManagerComponent:
    """
    Integrate auto-save functionality with a specific page.

    Args:
        page_name: Name of the page for identification
        config_data: Configuration data from the page

    Returns:
        AutoSaveManagerComponent instance
    """
    # Create page-specific auto-save manager
    auto_save_config = AutoSaveConfig(
        save_interval=5.0,  # Save every 5 seconds for pages
        debounce_interval=1.0,  # Wait 1 second after last change
        visual_feedback=True,
    )

    manager_component = create_auto_save_manager(auto_save_config)

    # Register the page configuration
    manager_component.integrate_with_config_form(config_data)

    return manager_component


# Global instance for shared use
_global_auto_save_manager: AutoSaveManagerComponent | None = None


def get_global_auto_save_manager() -> AutoSaveManagerComponent:
    """
    Get the global auto-save manager component.

    Returns:
        Global AutoSaveManagerComponent instance
    """
    global _global_auto_save_manager
    if _global_auto_save_manager is None:
        _global_auto_save_manager = create_auto_save_manager()
    return _global_auto_save_manager
