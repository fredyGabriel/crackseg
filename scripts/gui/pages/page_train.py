"""
Training page for CrackSeg application.

This module provides the user interface for training configuration and
monitoring. Contains comprehensive controls for model training, including
device selection, parameter configuration, and progress monitoring.
"""

import logging
from typing import Any

import streamlit as st
import torch

from scripts.gui.components.device_selector import device_selector
from scripts.gui.components.progress_bar_optimized import OptimizedProgressBar
from scripts.gui.components.tensorboard.component import TensorBoardComponent
from scripts.gui.services.gpu_monitor import GPUMonitor
from scripts.gui.utils.error_state import ErrorMessageFactory, ErrorType
from scripts.gui.utils.training_state import TrainingState

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize project root
PROJECT_ROOT = (
    "/c/Users/fgrv/OneDrive/Documentos/PythonProjects/doctorado/crackseg"
)

# // ... existing code ...


def render_training_configuration(state: dict[str, Any]) -> dict[str, Any]:
    """Render the training configuration form."""
    with st.expander("Training Configuration", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Basic Parameters")

            # Device selection using our new component
            st.write("### Device Selection")
            selected_device = device_selector(
                selected_device=state.get("selected_device"),
                component_id="training_device_selector",
                session_key="training_device",
                show_title=False,
            )

            # Update state with selected device
            state["selected_device"] = selected_device

            # Display device info
            if selected_device:
                device_info = st.empty()
                try:
                    device = torch.device(selected_device)
                    if device.type == "cuda":
                        memory_allocated = (
                            torch.cuda.memory_allocated(device) / 1024**3
                        )
                        memory_reserved = (
                            torch.cuda.memory_reserved(device) / 1024**3
                        )
                        memory_cached = (
                            torch.cuda.memory_cached(device) / 1024**3
                        )

                        device_info.info(
                            f"🎯 **Selected Device:** {selected_device}\n"
                            f"📊 **Memory Usage:** {memory_allocated:.2f} GB allocated\n"
                            f"📦 **Reserved:** {memory_reserved:.2f} GB\n"
                            f"📋 **Cached:** {memory_cached:.2f} GB"
                        )
                    else:
                        device_info.info(
                            f"🎯 **Selected Device:** {selected_device} (CPU)"
                        )
                except Exception as e:
                    device_info.warning(f"⚠️ Device info unavailable: {e}")

            # Epochs
            epochs = st.number_input(
                "Number of Epochs",
                min_value=1,
                max_value=1000,
                value=state.get("epochs", 50),
                step=1,
                help="Number of training epochs",
            )
            state["epochs"] = epochs

            # Learning Rate
            learning_rate = st.number_input(
                "Learning Rate",
                min_value=1e-6,
                max_value=1e-1,
                value=state.get("learning_rate", 1e-3),
                step=1e-5,
                format="%.2e",
                help="Initial learning rate for the optimizer",
            )
            state["learning_rate"] = learning_rate

            # Batch Size
            batch_size = st.number_input(
                "Batch Size",
                min_value=1,
                max_value=64,
                value=state.get("batch_size", 4),
                step=1,
                help="Batch size for training",
            )
            state["batch_size"] = batch_size

        with col2:
            st.subheader("Advanced Parameters")

            # Model Architecture
            architecture = st.selectbox(
                "Model Architecture",
                options=["UNet", "UNet++", "DeepLabV3+", "PSPNet"],
                index=0,
                help="Choose the model architecture",
            )
            state["architecture"] = architecture

            # Loss Function
            loss_function = st.selectbox(
                "Loss Function",
                options=["CrossEntropy", "Dice", "Focal", "Combined"],
                index=0,
                help="Choose the loss function",
            )
            state["loss_function"] = loss_function

            # Optimizer
            optimizer = st.selectbox(
                "Optimizer",
                options=["Adam", "SGD", "AdamW", "RMSprop"],
                index=0,
                help="Choose the optimizer",
            )
            state["optimizer"] = optimizer

            # Early Stopping
            early_stopping = st.checkbox(
                "Enable Early Stopping",
                value=state.get("early_stopping", True),
                help="Stop training early if validation loss doesn't improve",
            )
            state["early_stopping"] = early_stopping

            if early_stopping:
                patience = st.number_input(
                    "Patience",
                    min_value=1,
                    max_value=50,
                    value=state.get("patience", 10),
                    step=1,
                    help="Number of epochs to wait before stopping",
                )
                state["patience"] = patience

    return state


# // ... existing code ...


def page_train(state: dict[str, Any]) -> None:
    """Main training page function."""
    st.title("🚀 Model Training")
    st.markdown(
        "Configure and monitor your crack segmentation model training."
    )

    # Initialize components
    gpu_monitor = GPUMonitor()
    training_state = TrainingState()
    tensorboard_component = TensorBoardComponent(
        log_dir=f"{PROJECT_ROOT}/outputs/logs"
    )

    # Render training configuration
    state = render_training_configuration(state)

    # // ... existing code ...

    # Training controls
    st.subheader("Training Controls")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        start_training = st.button(
            "▶️ Start Training",
            type="primary",
            disabled=training_state.is_running,
            help="Start the training process",
        )

    with col2:
        pause_training = st.button(
            "⏸️ Pause Training",
            disabled=not training_state.is_running,
            help="Pause the current training",
        )

    with col3:
        stop_training = st.button(
            "⏹️ Stop Training",
            disabled=not training_state.is_running,
            help="Stop the training process",
        )

    with col4:
        resume_training = st.button(
            "▶️ Resume Training",
            disabled=not training_state.is_paused,
            help="Resume paused training",
        )

    # Handle training actions
    if start_training:
        try:
            # Validate device availability
            if state["selected_device"]:
                device = torch.device(state["selected_device"])
                if device.type == "cuda" and not torch.cuda.is_available():
                    st.error("CUDA is not available on this system!")
                    return

            # Start training with selected device
            training_state.start_training()
            st.success(
                f"Training started on device: {state['selected_device']}"
            )

            # Log training start
            logger.info(
                f"Training started with device: {state['selected_device']}"
            )

        except Exception as e:
            error_msg = ErrorMessageFactory.create_error_message(
                ErrorType.SYSTEM_ERROR,
                f"Failed to start training: {str(e)}",
            )
            st.error(error_msg)
            logger.error(f"Training start failed: {e}")

    if pause_training:
        training_state.pause_training()
        st.warning("Training paused.")

    if stop_training:
        training_state.stop_training()
        st.info("Training stopped.")

    if resume_training:
        training_state.resume_training()
        st.success("Training resumed.")

    # // ... existing code ...

    # Display training progress
    if training_state.is_running or training_state.is_paused:
        st.subheader("Training Progress")

        # Progress bar
        progress_bar = OptimizedProgressBar(0, "Training Progress")

        # Display metrics
        if training_state.current_metrics:
            col1, col2, col3 = st.columns(3)
            with col1:
                loss_delta = training_state.current_metrics.get(
                    "train_loss_delta", 0
                )
                st.metric(
                    "Training Loss",
                    f"{training_state.current_metrics.get('train_loss', 0):.4f}",
                    delta=loss_delta,
                )
            with col2:
                val_loss_delta = training_state.current_metrics.get(
                    "val_loss_delta", 0
                )
                st.metric(
                    "Validation Loss",
                    f"{training_state.current_metrics.get('val_loss', 0):.4f}",
                    delta=val_loss_delta,
                )
            with col3:
                accuracy_delta = training_state.current_metrics.get(
                    "accuracy_delta", 0
                )
                st.metric(
                    "Accuracy",
                    f"{training_state.current_metrics.get('accuracy', 0):.2%}",
                    delta=accuracy_delta,
                )

    # // ... existing code ...

    # TensorBoard integration
    st.subheader("TensorBoard Monitoring")

    # TensorBoard display
    tensorboard_component.render()

    # GPU monitoring
    if state.get("selected_device", "").startswith("cuda"):
        st.subheader("GPU Monitoring")

        # Display GPU information
        gpu_info = gpu_monitor.get_gpu_info()
        if gpu_info:
            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "GPU Memory Usage",
                    f"{gpu_info.get('memory_used', 0):.1f} GB",
                    f"/{gpu_info.get('memory_total', 0):.1f} GB",
                )

            with col2:
                st.metric(
                    "GPU Utilization",
                    f"{gpu_info.get('utilization', 0):.0f}%",
                )

            # Memory usage chart
            if gpu_info.get("memory_history"):
                st.line_chart(gpu_info["memory_history"])

    # // ... existing code ...
