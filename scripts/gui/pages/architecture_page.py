"""
Architecture page for the CrackSeg application.

This module contains the architecture visualization page that allows users to:
- Load model configurations
- Instantiate models from configurations
- Generate and display architecture diagrams
- View model summaries and statistics

Includes proper error handling for Graphviz installation and model
instantiation.
"""

import logging
from pathlib import Path
from typing import Any

import streamlit as st
import torch

from scripts.gui.components.loading_spinner import LoadingSpinner
from scripts.gui.components.progress_bar import (
    create_step_progress,
)
from scripts.gui.utils.architecture_viewer import (
    ArchitectureViewerError,
    GraphvizNotInstalledError,
    ModelInstantiationError,
    display_graphviz_installation_help,
    format_model_summary,
    get_architecture_viewer,
)
from scripts.gui.utils.config import (
    get_config_metadata,
    scan_config_directories,
)
from scripts.gui.utils.session_state import SessionStateManager

logger = logging.getLogger(__name__)


def page_architecture() -> None:
    """Architecture visualization page content."""
    st.header("🏗️ Model Architecture Viewer")

    # Get session state
    state = SessionStateManager.get()

    # Configuration selection section
    _render_configuration_selection()

    # Model instantiation section
    if state.config_path:
        _render_model_instantiation_section()
    else:
        st.info("📁 Please select a configuration file to proceed")
        return

    # Architecture visualization section
    if state.model_loaded and state.current_model is not None:
        _render_architecture_visualization_section()

    # Model information section
    if state.model_loaded and state.current_model is not None:
        _render_model_information_section()


def _render_configuration_selection() -> None:
    """Render the configuration file selection interface."""
    st.subheader("📋 Configuration Selection")

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


def _render_model_instantiation_section() -> None:
    """Render the model instantiation interface."""
    st.subheader("🔧 Model Instantiation")

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
            "🚀 Load Model",
            key="instantiate_model",
            help="Instantiate model from selected configuration",
            use_container_width=True,
        )

    with col3:
        # Clear model button
        if state.model_loaded:
            clear_button = st.button(
                "🗑️ Clear Model",
                help="Clear current model from memory",
                use_container_width=True,
            )
            if clear_button:
                _clear_model_state()
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
            f"✅ Model loaded: {state.model_architecture} "
            f"on {state.model_device}"
        )
    elif state.config_path:
        st.info("📋 Configuration loaded - ready to instantiate model")


def _instantiate_model_with_progress(
    state: Any, selected_device: str, viewer: Any
) -> None:
    """Instantiate model with progress tracking for complex models."""
    # Use LoadingSpinner for quick initialization
    message, subtext, spinner_type = LoadingSpinner.get_contextual_message(
        "model"
    )

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
            # Simulate config loading time
            import time

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

        st.success(f"✅ Model successfully loaded on {selected_device}")
        st.rerun()

    except GraphvizNotInstalledError:
        # This shouldn't happen during instantiation, but handle it
        # just in case
        st.warning("⚠️ Graphviz not available for visualization")
        logger.warning("Graphviz not available")

    except ModelInstantiationError as e:
        st.error(f"❌ Model instantiation failed: {e}")
        logger.error(f"Model instantiation error: {e}", exc_info=True)

    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        logger.error(f"Unexpected instantiation error: {e}", exc_info=True)


def _render_architecture_visualization_section() -> None:
    """Render the architecture visualization interface."""
    st.subheader("🎨 Architecture Visualization")

    state = SessionStateManager.get()
    viewer = get_architecture_viewer()

    # Check if Graphviz is available
    if not viewer.check_graphviz_available():
        display_graphviz_installation_help()
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        generate_button = st.button(
            "🎯 Generate Diagram",
            help="Generate architecture visualization diagram",
            use_container_width=True,
        )

    with col2:
        auto_generate = st.checkbox(
            "Auto-generate on model load",
            value=False,
            help="Automatically generate diagram when model is loaded",
        )

    # Diagram generation logic
    should_generate = generate_button or (
        auto_generate
        and state.model_loaded
        and state.architecture_diagram_path is None
    )

    if should_generate and state.current_model is not None:
        # Use contextual message for diagram generation
        diagram_message = "Generating architecture diagram..."
        diagram_subtext = "Creating visual representation of model structure"

        try:
            with LoadingSpinner.spinner(
                message=diagram_message,
                subtext=diagram_subtext,
                spinner_type="ai_processing",
                timeout_seconds=20,
            ):
                # Generate diagram
                diagram_path = viewer.generate_architecture_diagram(
                    state.current_model,
                    filename=f"architecture_{state.model_architecture}",
                )

                # Store path in session state
                state.architecture_diagram_path = str(diagram_path)

                SessionStateManager.notify_change("diagram_generated")

            st.success("✅ Architecture diagram generated successfully")

        except GraphvizNotInstalledError:
            display_graphviz_installation_help()
            return

        except ArchitectureViewerError as e:
            st.error(f"❌ Diagram generation failed: {e}")
            logger.error(f"Diagram generation error: {e}", exc_info=True)

        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
            logger.error(f"Unexpected diagram error: {e}", exc_info=True)

    # Display generated diagram
    if state.architecture_diagram_path:
        diagram_path = Path(state.architecture_diagram_path)
        if diagram_path.exists():
            st.markdown("### 📊 Architecture Diagram")
            st.image(
                str(diagram_path),
                caption=f"Architecture: {state.model_architecture}",
                use_column_width=True,
            )
        else:
            st.warning("Diagram file not found - please regenerate")


def _render_model_information_section() -> None:
    """Render the model information and statistics section."""
    st.subheader("📊 Model Information")

    state = SessionStateManager.get()

    if not state.model_summary:
        st.info("No model summary available")
        return

    # Create tabs for different information sections
    tab1, tab2, tab3 = st.tabs(["📈 Summary", "🔧 Components", "💾 Details"])

    with tab1:
        # Model summary
        summary_text = format_model_summary(state.model_summary)
        st.markdown(summary_text)

        # Parameter breakdown
        if "total_params" in state.model_summary:
            total = state.model_summary["total_params"]
            trainable = state.model_summary.get("trainable_params", total)
            frozen = total - trainable

            st.markdown("### Parameter Distribution")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total", f"{total:,}")
            with col2:
                st.metric("Trainable", f"{trainable:,}")
            with col3:
                st.metric("Frozen", f"{frozen:,}")

    with tab2:
        # Architecture components
        components = {}
        for key in [
            "encoder_type",
            "bottleneck_type",
            "decoder_type",
            "final_activation_type",
        ]:
            if key in state.model_summary:
                component_name = (
                    key.replace("_type", "").replace("_", " ").title()
                )
                components[component_name] = state.model_summary[key]

        if components:
            for component, type_name in components.items():
                st.markdown(f"**{component}:** {type_name}")
        else:
            st.info("Component information not available")

    with tab3:
        # Detailed information
        st.json(state.model_summary)


def _clear_model_state() -> None:
    """Clear all model-related state."""
    state = SessionStateManager.get()

    # Clear model state
    state.model_loaded = False
    state.current_model = None
    state.model_summary = {}
    state.model_device = None
    state.model_architecture = None

    # Clear diagram state
    state.architecture_diagram_path = None

    # Cleanup temporary files
    viewer = get_architecture_viewer()
    viewer.cleanup_temp_files()

    SessionStateManager.notify_change("model_cleared")
    logger.info("Model state cleared")
