"""
File Browser Component for Streamlit This module provides a reusable
file browser component to navigate and select files within a Streamlit
application.
"""

from collections.abc import Callable
from pathlib import Path

import streamlit as st


class FileBrowser:
    """
    A simple file browser component to navigate directories and select a
    file. This component maintains its own state (current directory)
    within Streamlit's session state, allowing it to be a reusable UI
    element. Attributes: root_dir (Path): The absolute path to the root
    directory for the browser. filter_glob (str): A glob pattern to filter
    the files displayed. on_select (Callable[[str], None] | None): A
    callback function that is triggered when a file is selected. It
    receives the absolute path of the selected file. key (str): A unique
    key to isolate the component's state in st.session_state.
    """

    def __init__(
        self,
        root_dir: str | Path = ".",
        filter_glob: str = "*",
        on_select: Callable[[str], None] | None = None,
        key: str = "file_browser",
    ):
        self.root_dir = Path(root_dir).resolve()
        self.filter_glob = filter_glob
        self.on_select = on_select
        self.key = key

        # Initialize the current directory in session state if not already
        # present.
        # This makes the component stateful across reruns.
        if f"{self.key}_current_dir" not in st.session_state:
            st.session_state[f"{self.key}_current_dir"] = str(self.root_dir)

    @property
    def current_dir(self) -> Path:
        """Get the current directory from session state."""
        return Path(st.session_state[f"{self.key}_current_dir"])

    def _set_current_dir(self, new_dir: Path) -> None:
        """Set the current directory in session state."""
        st.session_state[f"{self.key}_current_dir"] = str(new_dir)

    def render(self) -> None:
        """
        Render the file browser UI. This includes the current path display, an
        'Up' button for navigation, and lists of directories and selectable
        files.
        """
        # Display the current path relative to the root for user orientation.
        display_path = self.current_dir.relative_to(self.root_dir)
        st.caption(f"Current Directory: `{display_path}`")

        # Navigation: 'Up' button to move to the parent directory.
        if self.current_dir != self.root_dir:
            if st.button("⬆️ Up", key=f"{self.key}_up_button"):
                self._set_current_dir(self.current_dir.parent)
                st.rerun()

        # List directories and files, handling potential permission errors.
        try:
            items = sorted(self.current_dir.iterdir())
            directories = [item for item in items if item.is_dir()]
            files = [
                item
                for item in items
                if item.is_file() and item.match(self.filter_glob)
            ]
        except PermissionError:
            st.error("Permission denied to access this directory.")
            return

        # Display clickable buttons for each directory to navigate into it.
        for directory in directories:
            # Using directory name in the key to ensure uniqueness.
            if st.button(
                f"📁 {directory.name}", key=f"{self.key}_dir_{directory.name}"
            ):
                self._set_current_dir(directory)
                st.rerun()

        # Display clickable buttons for each file to trigger the on_select
        # callback.
        for file in files:
            if st.button(
                f"📄 {file.name}", key=f"{self.key}_file_{file.name}"
            ):
                if self.on_select:
                    # Pass the absolute path of the file to the callback.
                    self.on_select(str(file.absolute()))
                else:
                    st.info(f"You selected: {file.absolute()}")
