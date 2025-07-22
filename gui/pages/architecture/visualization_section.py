"""
Architecture visualization section for the architecture page. This
module handles architecture diagram generation and display, including
Graphviz integration and error handling.
"""

import logging
from pathlib import Path

import streamlit as st

from gui.components.loading_spinner import LoadingSpinner
from gui.utils.architecture_viewer import (
    ArchitectureViewerError,
    GraphvizNotInstalledError,
    display_graphviz_installation_help,
    get_architecture_viewer,
)
from gui.utils.session_state import SessionStateManager

logger = logging.getLogger(__name__)


def render_architecture_visualization_section() -> None:
    """Render the architecture visualization interface."""
    st.subheader("Architecture Visualization")

    state = SessionStateManager.get()
    viewer = get_architecture_viewer()

    # Check if Graphviz is available
    if not viewer.check_graphviz_available():
        display_graphviz_installation_help()
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        generate_button = st.button(
            "Generate Diagram",
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

            st.success("Architecture diagram generated successfully")

        except GraphvizNotInstalledError:
            display_graphviz_installation_help()
            return

        except ArchitectureViewerError as e:
            st.error(f"Diagram generation failed: {e}")
            logger.error(f"Diagram generation error: {e}", exc_info=True)

        except Exception as e:
            st.error(f"Unexpected error: {e}")
            logger.error(f"Unexpected diagram error: {e}", exc_info=True)

    # Display generated diagram
    if state.architecture_diagram_path:
        diagram_path = Path(state.architecture_diagram_path)
        if diagram_path.exists():
            st.markdown("### Architecture Diagram")
            st.image(
                str(diagram_path),
                caption=f"Architecture: {state.model_architecture}",
                use_column_width=True,
            )
        else:
            st.warning("Diagram file not found - please regenerate")
