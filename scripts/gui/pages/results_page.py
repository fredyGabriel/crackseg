"""
Results page for the CrackSeg application.

This module contains the results visualization page content for analyzing
predictions, displaying triplet galleries, and exporting reports with
professional reactive interface integration.
"""

from pathlib import Path
from typing import Any

import streamlit as st

from scripts.gui.components.results_gallery_component import (
    ResultsGalleryComponent,
)
from scripts.gui.components.tensorboard_component import TensorBoardComponent
from scripts.gui.utils.results import ValidationLevel
from scripts.gui.utils.session_state import SessionStateManager


def page_results() -> None:
    """Professional results visualization page with reactive gallery."""
    state = SessionStateManager.get()

    # Page header
    st.title("📊 Prediction Results & Analysis")
    st.markdown("---")

    # Check readiness
    if not state.is_ready_for_results():
        _render_setup_guide()
        return

    # Main results interface
    _render_results_interface(state)


def _render_setup_guide() -> None:
    """Render setup guide for users who haven't completed training."""
    st.warning("⚠️ **Setup Required**")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        ### 🚀 **Get Started**

        To access results visualization, you need:

        1. **Complete Training** - Train a model first
        2. **Set Run Directory** - Configure output directory
        3. **Generate Predictions** - Run inference on test data

        """
        )

    with col2:
        st.markdown(
            """
        ### 📋 **Quick Actions**

        - 🏃‍♂️ **[Training Page](/training)** - Start model training
        - ⚙️ **[Configuration](/config)** - Set up directories
        - 📖 **[Documentation](docs/)** - View training guide

        """
        )

    st.info(
        """
    💡 **Tip**: Once you have completed training, return to this page to:
    - View TensorBoard metrics
    - Browse prediction triplets (Image | Mask | Prediction)
    - Export results in various formats
    - Compare model performance
    """
    )


def _render_results_interface(state: Any) -> None:
    """Render the main results interface with tabs."""
    # Get run directory for scans
    run_dir = getattr(state, "run_dir", None)
    predictions_dir = None

    if run_dir:
        # Look for common prediction directories
        run_path = Path(run_dir)
        possible_pred_dirs = [
            run_path / "predictions",
            run_path / "outputs" / "predictions",
            run_path / "results",
            run_path / "inference",
        ]

        for pred_dir in possible_pred_dirs:
            if pred_dir.exists():
                predictions_dir = pred_dir
                break

    # Configuration panel
    with st.expander("⚙️ Configuration", expanded=False):
        _render_config_panel(predictions_dir)

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🖼️ Results Gallery",
            "📊 TensorBoard",
            "📈 Metrics Analysis",
            "🔄 Model Comparison",
        ]
    )

    with tab1:
        _render_gallery_tab(predictions_dir)

    with tab2:
        _render_tensorboard_tab(state)

    with tab3:
        _render_metrics_tab(state)

    with tab4:
        _render_comparison_tab(state)


def _render_config_panel(predictions_dir: Path | None) -> None:
    """Render configuration panel for results display."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📁 **Scan Directory**")
        if predictions_dir:
            st.success(f"✅ Auto-detected: `{predictions_dir}`")
        else:
            st.warning("⚠️ No predictions directory found")

        # Manual directory input
        custom_dir = st.text_input(
            "Custom Directory",
            value=str(predictions_dir) if predictions_dir else "",
            help="Override auto-detected predictions directory",
        )

        if custom_dir and Path(custom_dir).exists():
            st.session_state["custom_predictions_dir"] = custom_dir
            st.success(f"✅ Custom directory set: `{custom_dir}`")
        elif custom_dir:
            st.error("❌ Directory does not exist")

    with col2:
        st.markdown("### 🔍 **Validation Settings**")

        validation_level = st.selectbox(
            "Validation Level",
            options=[level.name for level in ValidationLevel],
            index=1,  # Default to STANDARD
            help="Choose validation thoroughness for triplet scanning",
        )
        st.session_state["global_validation_level"] = ValidationLevel[
            validation_level
        ]

        max_triplets = st.number_input(
            "Max Triplets",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            help="Maximum number of triplets to display",
        )
        st.session_state["global_max_triplets"] = max_triplets

    with col3:
        st.markdown("### 🎨 **Display Settings**")

        grid_columns = st.selectbox(
            "Grid Columns",
            options=[2, 3, 4, 5],
            index=1,  # Default to 3
            help="Number of columns in gallery grid",
        )
        st.session_state["global_grid_columns"] = grid_columns

        enable_real_time = st.checkbox(
            "Real-time Updates",
            value=True,
            help="Enable real-time gallery updates during scanning",
        )
        st.session_state["global_real_time"] = enable_real_time


def _render_gallery_tab(predictions_dir: Path | None) -> None:
    """Render the results gallery tab with triplet visualization."""
    # Determine scan directory
    scan_directory = None

    if st.session_state.get("custom_predictions_dir"):
        scan_directory = st.session_state["custom_predictions_dir"]
    elif predictions_dir:
        scan_directory = predictions_dir

    # Get settings from session state
    validation_level = st.session_state.get(
        "global_validation_level", ValidationLevel.STANDARD
    )
    max_triplets = st.session_state.get("global_max_triplets", 50)
    grid_columns = st.session_state.get("global_grid_columns", 3)
    enable_real_time = st.session_state.get("global_real_time", True)

    # Create and render gallery component
    gallery = ResultsGalleryComponent()

    gallery.render(
        scan_directory=scan_directory,
        validation_level=validation_level,
        max_triplets=max_triplets,
        grid_columns=grid_columns,
        show_validation_panel=True,
        show_export_panel=True,
        enable_real_time_scanning=enable_real_time,
    )

    gallery_state = gallery.ui_state

    # Display gallery statistics in sidebar
    if gallery_state["total_triplets"] > 0:
        with st.sidebar:
            st.markdown("### 📊 **Gallery Stats**")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total", gallery_state["total_triplets"])
                st.metric("Valid", gallery_state["valid_triplets"])

            with col2:
                st.metric("Selected", len(gallery_state["selected_triplets"]))
                cache_stats = gallery_state["cache_stats"]
                st.metric(
                    "Cache Hit", f"{cache_stats.get('hit_rate', 0):.1f}%"
                )

            # Quick actions
            st.markdown("### ⚡ **Quick Actions**")

            if st.button("🔄 Refresh Gallery", use_container_width=True):
                # Clear cache and rerun
                st.cache_data.clear()
                st.rerun()

            if gallery_state["total_triplets"] > 0:
                if st.button("📊 View Summary", use_container_width=True):
                    _show_gallery_summary(gallery_state)


def _render_tensorboard_tab(state: Any) -> None:
    """Render TensorBoard tab with integrated iframe embedding."""
    run_dir = getattr(state, "run_dir", None)

    if run_dir is None:
        st.info(
            "📂 No active run directory. "
            "Please configure a run directory first."
        )
        return

    # Construct log directory path
    log_dir = Path(run_dir) / "logs" / "tensorboard"

    # Create and render TensorBoard component
    tb_component = TensorBoardComponent(
        default_height=700,
        auto_startup=True,
        show_controls=True,
        show_status=True,
    )

    # Render with custom title
    tb_component.render(
        log_dir=log_dir,
        title="📊 TensorBoard Training Visualization",
        show_refresh=True,
    )


def _render_metrics_tab(state: Any) -> None:
    """Render metrics analysis tab."""
    st.subheader("📈 Training Metrics Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 **Performance Summary**")

        if state.last_evaluation:
            # Display evaluation metrics
            eval_data = state.last_evaluation

            # Create metrics display
            metrics_container = st.container()
            with metrics_container:
                metric_cols = st.columns(3)

                # Extract common metrics
                if "iou" in eval_data:
                    with metric_cols[0]:
                        st.metric("IoU Score", f"{eval_data['iou']:.4f}")

                if "dice" in eval_data:
                    with metric_cols[1]:
                        st.metric("Dice Score", f"{eval_data['dice']:.4f}")

                if "loss" in eval_data:
                    with metric_cols[2]:
                        st.metric(
                            "Validation Loss", f"{eval_data['loss']:.4f}"
                        )

            # Display full evaluation data
            st.markdown("### 📋 **Detailed Metrics**")
            st.json(eval_data)

        else:
            st.info(
                "No evaluation metrics available. "
                "Complete training to see results."
            )

    with col2:
        st.markdown("### 📊 **Metrics Visualization**")
        st.info("📈 Advanced metrics visualization coming soon!")

        st.markdown(
            """
        **Planned Features:**
        - 📉 Loss curves over time
        - 📊 Metric comparisons across epochs
        - 🎯 Performance distribution plots
        - 📈 Learning rate scheduling visualization
        """
        )


def _render_comparison_tab(state: Any) -> None:
    """Render model comparison tab."""
    st.subheader("🔄 Model Comparison")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 🏆 **Model Performance Comparison**")
        st.info("🔄 Model comparison tools coming soon!")

        st.markdown(
            """
        **Planned Features:**
        - 📊 Side-by-side metric comparison
        - 🖼️ Visual prediction comparisons
        - 📈 Performance trend analysis
        - 🎯 Best model recommendations
        - 📋 Automated evaluation reports
        """
        )

    with col2:
        st.markdown("### ⚙️ **Quick Setup**")

        if st.button("🔄 Scan Models", use_container_width=True):
            st.info("Model scanning functionality will be implemented soon.")

        if st.button("📊 Generate Report", use_container_width=True):
            st.info(
                "Report generation functionality will be implemented soon."
            )

        st.markdown("### 📁 **Model Registry**")
        st.info("No models registered yet.")


def _show_gallery_summary(gallery_state: dict[str, Any]) -> None:
    """Show gallery summary in expandable dialog."""
    with st.expander("📊 Gallery Summary", expanded=True):
        st.markdown("### 🎯 **Scan Results**")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Triplets", gallery_state["total_triplets"])

        with col2:
            st.metric("Valid Triplets", gallery_state["valid_triplets"])

        with col3:
            success_rate = (
                gallery_state["valid_triplets"]
                / gallery_state["total_triplets"]
                * 100.0
                if gallery_state["total_triplets"] > 0
                else 0.0
            )
            st.metric("Success Rate", f"{success_rate:.1f}%")

        st.markdown("### 🗂️ **Selection Summary**")
        selected_count = len(gallery_state["selected_triplets"])
        st.write(f"Selected: **{selected_count}** triplets")

        if selected_count > 0:
            st.markdown("**Selected Items:**")
            for i, triplet in enumerate(
                gallery_state["selected_triplets"][:10]
            ):
                st.write(f"{i + 1}. {triplet.id} ({triplet.dataset_name})")

            if selected_count > 10:
                st.write(f"... and {selected_count - 10} more")

        # Cache performance
        cache_stats = gallery_state["cache_stats"]
        st.markdown("### ⚡ **Cache Performance**")

        cache_col1, cache_col2 = st.columns(2)
        with cache_col1:
            st.metric("Hit Rate", f"{cache_stats.get('hit_rate', 0):.1f}%")
        with cache_col2:
            st.metric("Cached Items", cache_stats.get("size", 0))
