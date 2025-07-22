"""
Configuration selection section for the architecture page. This module
handles the configuration file selection interface, allowing users to
browse and select model architecture configurations.
"""

import logging
from pathlib import Path

import streamlit as st

from gui.utils.config import (
    get_config_metadata,
    scan_config_directories,
)
from gui.utils.session_state import SessionStateManager

logger = logging.getLogger(__name__)


def render_configuration_selection() -> None:
    """Render the configuration file selection interface."""
    st.subheader("Configuration Selection")

    state = SessionStateManager.get()

    col1, col2 = st.columns([3, 1])

    with col1:
        try:
            # Scan for available configurations
            config_dirs = scan_config_directories()

            # Get architecture configurations
            arch_configs = []
            if "configs/model" in config_dirs:
                for config_path_str in config_dirs["configs/model"]:
                    config_path = Path(config_path_str)
                    if "architectures" in config_path.parts:
                        # Get metadata for this config file
                        metadata = get_config_metadata(config_path)
                        arch_configs.append(metadata)

            if not arch_configs:
                st.warning("No architecture configuration files found")
                return

            # Create options for selectbox
            config_options = [
                f"{Path(info['path']).name} "
                f"({info.get('size_human', 'unknown size')})"
                for info in arch_configs
            ]

            # Add current selection if available
            current_index = 0
            if state.config_path:
                for i, info in enumerate(arch_configs):
                    if Path(str(info["path"])) == Path(state.config_path):
                        current_index = i
                        break

            selected_option = st.selectbox(
                "Select Model Configuration:",
                config_options,
                index=current_index,
                help="Choose a model architecture configuration file",
            )

            # Update session state
            if selected_option:
                selected_index = config_options.index(selected_option)
                selected_config = arch_configs[selected_index]
                config_path_str = str(selected_config["path"])
                if state.config_path != config_path_str:
                    state.config_path = config_path_str
                    # Clear model state when config changes
                    state.model_loaded = False
                    state.current_model = None
                    state.model_summary = {}
                    state.architecture_diagram_path = None
                    SessionStateManager.notify_change("config_changed")

        except Exception as e:
            st.error(f"Error scanning configurations: {e}")
            logger.error(f"Configuration scanning error: {e}", exc_info=True)

    with col2:
        # Display current config info
        if state.config_path:
            config_path = Path(state.config_path)
            st.info(f"**Selected:**\n{config_path.name}")
        else:
            st.info("No config selected")
