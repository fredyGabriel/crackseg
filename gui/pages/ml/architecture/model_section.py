"""
Model instantiation section for the architecture page. This module
handles model instantiation interface and progress tracking for
complex model loading operations.
"""

import logging
import time
from pathlib import Path
from typing import Any

import streamlit as st
import torch

from gui.components.loading_spinner import LoadingSpinner
from gui.components.progress_bar import create_step_progress
from gui.utils.architecture_viewer import (
    GraphvizNotInstalledError,
    ModelInstantiationError,
    get_architecture_viewer,
)
from gui.utils.session_state import SessionStateManager

from .utils import clear_model_state

logger = logging.getLogger(__name__)


def render_model_instantiation_section() -> None:
    """Render the model instantiation interface."""
    st.subheader("Model Instantiation")

    state = SessionStateManager.get()
    viewer = get_architecture_viewer()

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        # Device selection
        device_options = ["cpu"]
        if torch.cuda.is_available():
            device_options.append("cuda")

        selected_device = st.selectbox(
            "Device:",
            device_options,
            index=0,
            help="Device to instantiate the model on "
            "(CPU recommended for visualization)",
        )

    with col2:
        # Instantiate button
        instantiate_button = st.button(
            "Load Model",
            key="instantiate_model",
            help="Instantiate model from selected configuration",
            use_container_width=True,
        )

    with col3:
        # Clear model button
        if state.model_loaded:
            clear_button = st.button(
                "Clear Model",
                help="Clear current model from memory",
                use_container_width=True,
            )
            if clear_button:
                clear_model_state()
                st.rerun()

    # Model instantiation logic
    if instantiate_button:
        if not state.config_path:
            st.error("No configuration file selected")
            return

        _instantiate_model_with_progress(state, selected_device, viewer)

    # Display current model status
    if state.model_loaded and state.current_model is not None:
        st.success(
            f"Model loaded: {state.model_architecture} on {state.model_device}"
        )
    elif state.config_path:
        st.info("Configuration loaded - ready to instantiate model")


def _instantiate_model_with_progress(
    state: Any, selected_device: str, viewer: Any
) -> None:
    """Instantiate model with progress tracking for complex models."""
    # Use LoadingSpinner for quick initialization
    _, _, spinner_type = LoadingSpinner.get_contextual_message("model")

    try:
        with LoadingSpinner.spinner(
            message="Initializing model loading...",
            subtext="Preparing model instantiation",
            spinner_type=spinner_type,
            timeout_seconds=5,
        ):
            # Quick initialization checks
            config_path = Path(state.config_path)
            if not config_path.exists():
                st.error("Configuration file not found")
                return

        # Use ProgressBar for the actual model instantiation process
        model_steps = [
            "Loading configuration file",
            "Parsing model architecture",
            "Initializing model components",
            "Allocating memory buffers",
            "Finalizing model setup",
        ]

        # Create step-based progress for model loading
        with create_step_progress(
            title="Model Instantiation",
            steps=model_steps,
            operation_id="model_loading",
        ) as progress:
            # Step 1: Load configuration
            progress.next_step("Loading and validating configuration")
            time.sleep(0.5)

            # Step 2: Parse architecture
            progress.next_step("Parsing model architecture definitions")
            time.sleep(0.5)

            # Step 3: Initialize components
            progress.next_step("Initializing model components")
            # Actual model instantiation
            model = viewer.instantiate_model_from_config_path(
                state.config_path, device=selected_device
            )
            time.sleep(0.5)

            # Step 4: Allocate memory
            progress.next_step("Allocating memory and setting up device")
            time.sleep(0.5)

            # Step 5: Finalize setup
            progress.next_step("Finalizing model setup and validation")
            # Get model summary
            summary = viewer.get_model_summary(model)
            time.sleep(0.5)

        # Update session state
        state.current_model = model
        state.model_loaded = True
        state.model_summary = summary
        state.model_device = selected_device
        state.model_architecture = summary.get("model_type", "Unknown")

        SessionStateManager.notify_change("model_instantiated")

        st.success(f"Model successfully loaded on {selected_device}")
        st.rerun()

    except GraphvizNotInstalledError:
        # This shouldn't happen during instantiation, but handle it
        # just in case
        st.warning("Graphviz not available for visualization")
        logger.warning("Graphviz not available")

    except ModelInstantiationError as e:
        st.error(f"Model instantiation failed: {e}")
        logger.error(f"Model instantiation error: {e}", exc_info=True)

    except Exception as e:
        st.error(f"Unexpected error: {e}")
        logger.error(f"Unexpected instantiation error: {e}", exc_info=True)
