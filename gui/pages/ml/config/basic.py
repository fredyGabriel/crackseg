"""
Configuration page for the CrackSeg application. This module contains
the configuration dashboard content for setting up experiments and
loading model configurations.
"""

from pathlib import Path

import streamlit as st

from gui.components.config_editor_component import (
    ConfigEditorComponent,
)
from gui.components.file_browser import FileBrowser
from gui.components.file_upload_component import FileUploadComponent
from gui.components.header_component import render_header
from gui.utils.gui_config import PAGE_CONFIG
from gui.utils.save_dialog import SaveDialogManager
from gui.utils.session_state import SessionStateManager


def page_config() -> None:
    """
    Renders the main configuration page for the application. This page is
    organized into collapsible sections for clarity: 1. **Model
    Configuration**: Load configs via file browser or upload. 2. **Editor
    & Validation**: Edit loaded configs and save changes. 3. **Output &
    Run Directory**: Set the destination for training outputs.
    """
    st.title(PAGE_CONFIG["Config"]["title"])
    state = SessionStateManager.get()
    render_header("Experiment Configuration")

    st.markdown(
        """
Configure your experiment by loading a YAML file, setting an output
directory, and validating the setup before training.
"""
    )

    # --- Expander 1: Model Configuration ---
    with st.expander("Model Configuration", expanded=not state.config_loaded):
        st.subheader("Load Configuration File")
        # Show current status
        if state.config_loaded:
            st.success("Configuration loaded")
            if state.config_path:
                st.caption(f"File: {Path(state.config_path).name}")
        else:
            st.warning("Configuration not loaded")

        tab1, tab2 = st.tabs(["Browse Project Files", "Upload from Computer"])

        with tab1:
            st.subheader("Select a Configuration from Project")

            def handle_file_select(file_path: str):
                """Callback function for when a file is selected."""
                absolute_path = str(Path(file_path).absolute())
                state.update_config(absolute_path, {"loaded": True})
                state.add_notification(
                    f"Configuration selected: {Path(file_path).name}"
                )
                st.rerun()

            # Instantiate and render the file browser
            browser = FileBrowser(
                root_dir="configs",
                filter_glob="*.yaml",
                on_select=handle_file_select,
                key="config_browser",
            )
            browser.render()

        with tab2:
            st.subheader("Upload a Configuration File")
            # File upload section
            upload_result = FileUploadComponent.render_upload_section(
                title="",
                help_text=(
                    "Upload a YAML file. It will be saved to "
                    "`generated_configs/` and loaded automatically."
                ),
                target_directory="generated_configs",
                key_suffix="_config_page",
                show_validation=True,
                show_preview=True,
            )

            if upload_result:
                st.rerun()

    # --- Expander 2: Editor & Validation ---
    with st.expander("Editor & Real-time Validation", expanded=True):
        if not state.config_loaded or not state.config_path:
            st.info(
                "Load a configuration file from the section above to enable "
                "the editor."
            )
        else:
            editor = ConfigEditorComponent()
            save_dialog = SaveDialogManager()

            # Read content from the loaded config file
            try:
                # Use a session state key to store content, preventing re-reads
                if (
                    "config_content" not in st.session_state
                    or st.session_state.get("config_path_loaded")
                    != state.config_path
                ):
                    st.session_state.config_content = Path(
                        state.config_path
                    ).read_text(encoding="utf-8")
                    st.session_state.config_path_loaded = state.config_path
            except Exception as e:
                st.error(f"Error reading configuration file: {e}")
                st.session_state.config_content = "# Error reading file."

            st.subheader("Edit Configuration")
            # Render the editor and get potentially modified content
            edited_content = editor.render_editor_with_advanced_validation(
                initial_content=st.session_state.config_content,
                key="main_editor",
            )
            st.session_state.config_content = edited_content

            # Save functionality
            st.markdown("---")
            st.subheader("Save Changes")
            if st.button("Save Configuration As..."):
                st.session_state.show_save_dialog = True

            if st.session_state.get("show_save_dialog", False):
                if save_dialog.render_save_dialog(
                    content=edited_content,
                    key="main_save_dialog",
                    default_name=Path(state.config_path).stem,
                ):
                    # If saved, hide dialog and update content
                    st.session_state.show_save_dialog = False
                    st.rerun()

    # --- Expander 3: Output Settings ---
    with st.expander(
        "Output & Run Directory", expanded=not state.run_directory
    ):
        st.subheader("Set Run Directory")
        # Show current status
        if state.run_directory:
            st.success("Directory configured")
            st.caption(f"Directory: {Path(state.run_directory).name}")
        else:
            st.warning("Directory not configured")

        st.markdown(
            """
**The run directory will store:** - Trained models and checkpoints -
Training logs and metrics - Evaluation results and visualizations
"""
        )

        run_dir_input = st.text_input(
            "Run Directory Path",
            key="run_dir_input",
            value=state.run_directory or "",
            placeholder="e.g., C:/Users/YourUser/crackseg_runs",
            help="Enter the full path to the desired output directory.",
        )
        if run_dir_input != state.run_directory:
            # Direct assignment instead of calling non-existent method
            state.run_directory = run_dir_input
            st.rerun()

    # --- Status Panel ---
    st.markdown("---")
    render_header("Setup Status")
    col1, col2, col3 = st.columns(3)

    with col1:
        # Configuration Status
        if state.config_loaded:
            st.success("Configuration: Loaded")
            # Handle potential None value for config_path
            config_name = "Unknown"
            if state.config_path:
                config_name = Path(state.config_path).name
            st.caption(f"File: {config_name}")
        else:
            st.error("Configuration: Not loaded")

    with col2:
        # Run Directory Status
        if state.run_directory:
            st.success("Output Directory: Set")
            st.caption(f"Path: {Path(state.run_directory).name}")
        else:
            st.error("Output Directory: Not set")

    with col3:
        # Overall Readiness - remove invalid return_issues parameter
        ready = state.is_ready_for_training()
        if ready:
            st.success("System Ready for Training")
        else:
            st.error("System Not Ready")
            st.caption("- Check configuration and directory settings")


# --- Theme Integration Section ---
def render_theme_controls() -> None:
    """Render theme selection controls for the configuration page."""
    st.markdown("---")
    render_header("Theme & Display Settings")

    # Theme component integration would be implemented here
    st.info("Theme controls will be available in future versions")
