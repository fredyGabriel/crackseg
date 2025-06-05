"""
File Browser Component for CrackSeg GUI.

This component provides an interface for browsing, selecting, and managing
YAML configuration files from the configs/ and generated_configs/ directories.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from scripts.gui.utils.config import (
    get_config_metadata,
    scan_config_directories,
)


class FileBrowserComponent:
    """File browser component for navigating and selecting YAML files."""

    def __init__(self) -> None:
        """Initialize the file browser component."""
        self.supported_extensions = {".yaml", ".yml"}
        self.sort_options = {
            "Name (A-Z)": "name_asc",
            "Name (Z-A)": "name_desc",
            "Modified (Newest)": "modified_desc",
            "Modified (Oldest)": "modified_asc",
            "Size (Largest)": "size_desc",
            "Size (Smallest)": "size_asc",
        }

    def render(
        self,
        key: str = "file_browser",
        show_preview: bool = True,
        allow_multiple: bool = False,
        filter_text: str = "",
    ) -> dict[str, Any]:
        """Render the file browser component.

        Args:
            key: Unique key for the component.
            show_preview: Whether to show file preview panel.
            allow_multiple: Whether to allow multiple file selection.
            filter_text: Text to filter files by name.

        Returns:
            Dictionary containing selected files and browser state.
        """
        # Initialize component state
        if f"{key}_state" not in st.session_state:
            st.session_state[f"{key}_state"] = {
                "selected_files": [],
                "current_directory": "configs",
                "sort_by": "name_asc",
                "show_hidden": False,
            }

        state = st.session_state[f"{key}_state"]

        # Create main container
        with st.container():
            # Header with controls
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                st.subheader("📁 Configuration Files")

            with col2:
                # Sort options
                sort_option = st.selectbox(
                    "Sort by:",
                    options=list(self.sort_options.keys()),
                    index=list(self.sort_options.values()).index(
                        state["sort_by"]
                    ),
                    key=f"{key}_sort",
                )
                state["sort_by"] = self.sort_options[sort_option]

            with col3:
                # Filter/search
                filter_input = st.text_input(
                    "🔍 Filter files:",
                    value=filter_text,
                    placeholder="Search files...",
                    key=f"{key}_filter",
                )

            # Directory navigation
            self._render_directory_nav(key, state)

            # File list
            col_files, col_preview = st.columns(
                [3, 2] if show_preview else [1]
            )

            with col_files:
                selected_files = self._render_file_list(
                    key, state, filter_input, allow_multiple
                )

            if show_preview:
                with col_preview:
                    self._render_file_preview(key, selected_files)

            # Update state
            state["selected_files"] = selected_files

        return {
            "selected_files": selected_files,
            "current_directory": state["current_directory"],
            "total_files": len(self._get_filtered_files(filter_input)),
        }

    def _render_directory_nav(self, key: str, state: dict[str, Any]) -> None:
        """Render directory navigation breadcrumbs and controls."""
        st.markdown("---")

        # Breadcrumb navigation
        col1, col2 = st.columns([3, 1])

        with col1:
            # Get available directories
            config_dirs = scan_config_directories()
            available_dirs = list(config_dirs.keys())

            if available_dirs:
                current_dir = st.selectbox(
                    "📂 Current Directory:",
                    options=available_dirs,
                    index=(
                        available_dirs.index(state["current_directory"])
                        if state["current_directory"] in available_dirs
                        else 0
                    ),
                    key=f"{key}_current_dir",
                )
                state["current_directory"] = current_dir
            else:
                st.warning("No configuration directories found.")
                st.info("Expected directories: configs/, generated_configs/")

        with col2:
            # Refresh button
            if st.button("🔄 Refresh", key=f"{key}_refresh"):
                # Clear any cached data
                st.cache_data.clear()

    def _render_file_list(
        self,
        key: str,
        state: dict[str, Any],
        filter_text: str,
        allow_multiple: bool,
    ) -> list[str]:
        """Render the file list with selection controls."""
        files = self._get_filtered_files(filter_text)
        sorted_files = self._sort_files(files, state["sort_by"])

        if not sorted_files:
            st.info("No YAML files found in the selected directory.")
            if filter_text:
                st.caption(f"No files match filter: '{filter_text}'")
            return []

        st.caption(f"Found {len(sorted_files)} YAML file(s)")

        selected_files: list[str] = []

        # Render file selection interface
        if allow_multiple:
            selected_files = self._render_multi_select(key, sorted_files)
        else:
            selected_file = self._render_single_select(key, sorted_files)
            if selected_file:
                selected_files = [selected_file]

        return selected_files

    def _render_multi_select(self, key: str, files: list[str]) -> list[str]:
        """Render multi-select file interface."""
        st.markdown("**Select files** (multiple selection enabled):")

        selected_files: list[str] = []

        # Add "Select All" / "Clear All" controls
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Select All", key=f"{key}_select_all"):
                st.session_state[f"{key}_multi_select"] = files
        with col2:
            if st.button("❌ Clear All", key=f"{key}_clear_all"):
                st.session_state[f"{key}_multi_select"] = []

        # Multi-select widget
        selected_files = st.multiselect(
            "Files:",
            options=files,
            default=st.session_state.get(f"{key}_multi_select", []),
            key=f"{key}_multi_select",
            format_func=lambda x: self._format_filename(x),
        )

        return selected_files

    def _render_single_select(self, key: str, files: list[str]) -> str | None:
        """Render single-select file interface."""
        st.markdown("**Select a file:**")

        # Radio button selection
        selected_file = st.radio(
            "Files:",
            options=[None] + files,
            index=0,
            key=f"{key}_single_select",
            format_func=lambda x: (
                "None selected" if x is None else self._format_filename(x)
            ),
        )

        return selected_file

    def _render_file_preview(
        self, key: str, selected_files: list[str]
    ) -> None:
        """Render file preview panel."""
        st.markdown("### 📄 File Preview")

        if not selected_files:
            st.info("Select a file to see preview")
            return

        # Show preview for first selected file
        file_path = selected_files[0]
        if len(selected_files) > 1:
            st.caption(f"Showing preview for: {Path(file_path).name}")
            st.caption(f"({len(selected_files)} files selected)")

        try:
            metadata = get_config_metadata(file_path)

            # File info
            st.markdown("**File Information:**")
            col1, col2 = st.columns(2)

            with col1:
                size_value = metadata.get("size_human", "Unknown")
                if isinstance(size_value, str):
                    st.metric("Size", size_value)
                else:
                    st.metric("Size", "Unknown")

            with col2:
                modified = metadata.get("modified")
                if modified and isinstance(modified, str):
                    try:
                        mod_date = datetime.fromisoformat(modified).strftime(
                            "%Y-%m-%d %H:%M"
                        )
                        st.metric("Modified", mod_date)
                    except ValueError:
                        st.metric("Modified", "Unknown")
                else:
                    st.metric("Modified", "Unknown")

            # File preview
            preview_lines = metadata.get("preview", [])
            if preview_lines and isinstance(preview_lines, list):
                st.markdown("**Preview (first 5 lines):**")
                preview_text = "\n".join(preview_lines)
                st.code(preview_text, language="yaml")
            else:
                st.warning("Unable to load file preview")

            # File actions
            st.markdown("**Actions:**")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("📝 Edit", key=f"{key}_edit_{hash(file_path)}"):
                    st.session_state.file_browser_action = "edit"
                    st.session_state.file_browser_target = file_path

            with col2:
                if st.button(
                    "✅ Validate", key=f"{key}_validate_{hash(file_path)}"
                ):
                    st.session_state.file_browser_action = "validate"
                    st.session_state.file_browser_target = file_path

        except Exception as e:
            st.error(f"Error loading file preview: {str(e)}")

    def _get_filtered_files(self, filter_text: str = "") -> list[str]:
        """Get list of YAML files filtered by search text."""
        try:
            config_dirs = scan_config_directories()
            all_files: list[str] = []

            for _category, files in config_dirs.items():
                all_files.extend(files)

            # Filter by extension
            yaml_files = [
                f
                for f in all_files
                if Path(f).suffix.lower() in self.supported_extensions
            ]

            # Apply text filter
            if filter_text:
                filter_lower = filter_text.lower()
                yaml_files = [
                    f
                    for f in yaml_files
                    if filter_lower in Path(f).name.lower()
                ]

            return yaml_files

        except Exception as e:
            st.error(f"Error scanning files: {str(e)}")
            return []

    def _sort_files(self, files: list[str], sort_by: str) -> list[str]:
        """Sort files according to the specified criteria."""
        try:
            if sort_by == "name_asc":
                return sorted(files, key=lambda f: Path(f).name.lower())
            elif sort_by == "name_desc":
                return sorted(
                    files, key=lambda f: Path(f).name.lower(), reverse=True
                )
            elif sort_by == "modified_desc":
                return sorted(
                    files,
                    key=lambda f: (
                        os.path.getmtime(f) if os.path.exists(f) else 0
                    ),
                    reverse=True,
                )
            elif sort_by == "modified_asc":
                return sorted(
                    files,
                    key=lambda f: (
                        os.path.getmtime(f) if os.path.exists(f) else 0
                    ),
                )
            elif sort_by == "size_desc":
                return sorted(
                    files,
                    key=lambda f: (
                        os.path.getsize(f) if os.path.exists(f) else 0
                    ),
                    reverse=True,
                )
            elif sort_by == "size_asc":
                return sorted(
                    files,
                    key=lambda f: (
                        os.path.getsize(f) if os.path.exists(f) else 0
                    ),
                )
            else:
                return files

        except Exception as e:
            st.error(f"Error sorting files: {str(e)}")
            return files

    def _format_filename(self, file_path: str) -> str:
        """Format filename for display in selection widgets."""
        path = Path(file_path)

        # Show relative path from project root if file is deeply nested
        try:
            # Get relative path from current working directory
            rel_path = path.relative_to(Path.cwd())
            if len(rel_path.parts) > 2:
                # Show parent folder + filename for nested files
                return f"{rel_path.parent.name}/{rel_path.name}"
            else:
                return rel_path.name
        except ValueError:
            # If file is outside current dir, just show filename
            return path.name

    @staticmethod
    def get_selected_action() -> tuple[str | None, str | None]:
        """Get the last action triggered from file preview.

        Returns:
            Tuple of (action, target_file_path) or (None, None).
        """
        action = st.session_state.get("file_browser_action")
        target = st.session_state.get("file_browser_target")

        if action and target:
            # Clear action after retrieving
            st.session_state.file_browser_action = None
            st.session_state.file_browser_target = None
            return action, target

        return None, None


# Convenience function for easy usage
def render_file_browser(
    key: str = "file_browser",
    show_preview: bool = True,
    allow_multiple: bool = False,
    filter_text: str = "",
) -> dict[str, Any]:
    """Render a file browser component.

    Args:
        key: Unique key for the component.
        show_preview: Whether to show file preview panel.
        allow_multiple: Whether to allow multiple file selection.
        filter_text: Text to filter files by name.

    Returns:
        Dictionary containing selected files and browser state.
    """
    browser = FileBrowserComponent()
    return browser.render(key, show_preview, allow_multiple, filter_text)
