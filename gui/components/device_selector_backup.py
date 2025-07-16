"""
Device selector component for the CrackSeg application.

This module provides a high-performance device selector with GPU/CPU detection,
detailed device information, and integration with the existing optimization
system. Allows users to choose between available devices for training
operations.
"""

import functools
import time
from typing import Any

import streamlit as st
import torch

from crackseg.utils.core.device import get_device
from crackseg.utils.logging import get_logger
from scripts.gui.utils.error_state import (
    ErrorInfo,
    ErrorType,
    StandardErrorState,
)
from scripts.gui.utils.performance_optimizer import (
    get_optimizer,
    inject_css_once,
)

logger = get_logger(__name__)


def track_performance_decorator(operation: str):
    """Decorator to track performance of component operations."""

    def decorator(func: Any) -> Any:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Track performance using the optimizer
                component_id = kwargs.get("component_id", "device_selector")
                get_optimizer().track_performance(
                    component_id, operation, start_time
                )

        return wrapper

    return decorator


class DeviceInfo:
    """Information about a compute device."""

    def __init__(
        self,
        device_id: str,
        device_type: str,
        device_name: str,
        memory_total: float | None = None,
        memory_available: float | None = None,
        compute_capability: str | None = None,
        is_available: bool = True,
    ) -> None:
        """Initialize device information."""
        self.device_id = device_id
        self.device_type = device_type
        self.device_name = device_name
        self.memory_total = memory_total
        self.memory_available = memory_available
        self.compute_capability = compute_capability
        self.is_available = is_available

    def to_dict(self) -> dict[str, Any]:
        """Convert device info to dictionary."""
        return {
            "device_id": self.device_id,
            "device_type": self.device_type,
            "device_name": self.device_name,
            "memory_total": self.memory_total,
            "memory_available": self.memory_available,
            "compute_capability": self.compute_capability,
            "is_available": self.is_available,
        }


class DeviceDetector:
    """Utility class for detecting available devices."""

    @staticmethod
    def get_available_devices() -> list[DeviceInfo]:
        """Get list of available devices."""
        devices = []

        # Add CPU device
        devices.append(DeviceInfo("cpu", "cpu", "CPU", is_available=True))

        # Add CUDA devices if available
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    device_name = torch.cuda.get_device_name(i)
                    props = torch.cuda.get_device_properties(i)

                    # Memory calculations
                    memory_total = props.total_memory / (1024**3)  # GB
                    memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                    memory_available = memory_total - memory_reserved

                    # Compute capability
                    compute_capability = f"{props.major}.{props.minor}"

                    devices.append(
                        DeviceInfo(
                            f"cuda:{i}",
                            "cuda",
                            device_name,
                            memory_total=memory_total,
                            memory_available=memory_available,
                            compute_capability=compute_capability,
                            is_available=True,
                        )
                    )
                except Exception:
                    # Device exists but might not be accessible
                    devices.append(
                        DeviceInfo(
                            f"cuda:{i}",
                            "cuda",
                            f"GPU {i}",
                            is_available=False,
                        )
                    )

        return devices

    @staticmethod
    def get_recommended_device() -> str:
        """Get recommended device based on availability and performance."""
        devices = DeviceDetector.get_available_devices()

        # Prefer CUDA devices with most memory
        cuda_devices = [
            d for d in devices if d.device_type == "cuda" and d.is_available
        ]
        if cuda_devices:
            # Sort by memory total (descending)
            cuda_devices.sort(
                key=lambda d: d.memory_total if d.memory_total else 0,
                reverse=True,
            )
            return cuda_devices[0].device_id

        # Fallback to CPU
        return "cpu"


class OptimizedDeviceSelector:
    """High-performance device selector with caching and optimization."""

    _CSS_CONTENT = """
    <style>
    .crackseg-device-selector {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        margin: 1rem 0;
    }

    .crackseg-device-selection-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .crackseg-device-selection-title {
        color: #2c3e50;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        text-align: center;
    }

    .crackseg-device-selection-subtitle {
        color: #7f8c8d;
        font-size: 0.95rem;
        text-align: center;
        margin-bottom: 1rem;
    }

    .crackseg-device-card {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        border: 2px solid #e3e3e3;
        transition: all 0.3s ease;
        cursor: pointer;
    }

    .crackseg-device-card:hover {
        border-color: #3498db;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(52, 152, 219, 0.2);
    }

    .crackseg-device-card.selected {
        border-color: #27ae60;
        background: linear-gradient(135deg, #a8e6cf 0%, #dcedc1 100%);
    }

    .crackseg-device-card.recommended {
        border-color: #f39c12;
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
    }

    .crackseg-device-card.unavailable {
        opacity: 0.6;
        background: #f8f9fa;
        border-color: #dee2e6;
    }

    .crackseg-device-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.75rem;
    }

    .crackseg-device-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2c3e50;
        margin: 0;
    }

    .crackseg-device-type {
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 500;
        text-transform: uppercase;
    }

    .crackseg-device-type.cpu {
        background: #95a5a6;
        color: white;
    }

    .crackseg-device-type.cuda {
        background: #76b900;
        color: white;
    }

    .crackseg-device-info {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.75rem;
        margin-bottom: 0.75rem;
    }

    .crackseg-device-info-item {
        display: flex;
        flex-direction: column;
    }

    .crackseg-device-info-label {
        font-size: 0.8rem;
        color: #7f8c8d;
        font-weight: 500;
        margin-bottom: 0.25rem;
    }

    .crackseg-device-info-value {
        font-size: 0.9rem;
        color: #2c3e50;
        font-weight: 600;
    }

    .crackseg-device-badges {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
    }

    .crackseg-device-badge {
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
    }

    .crackseg-device-badge.recommended {
        background: #f39c12;
        color: white;
    }

    .crackseg-device-badge.selected {
        background: #27ae60;
        color: white;
    }

    .crackseg-device-badge.unavailable {
        background: #e74c3c;
        color: white;
    }

    @media (max-width: 768px) {
        .crackseg-device-info {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """

    _BRAND_COLORS = {
        "primary": "#3498db",
        "secondary": "#2c3e50",
        "accent": "#e74c3c",
        "success": "#27ae60",
        "warning": "#f39c12",
        "error": "#e74c3c",
        "info": "#17a2b8",
        "cpu": "#95a5a6",
        "cuda": "#76b900",
    }

    @staticmethod
    def _ensure_css_injected() -> None:
        """Ensure CSS is injected only once per session for performance."""
        inject_css_once(
            "crackseg_device_selector", OptimizedDeviceSelector._CSS_CONTENT
        )

    @staticmethod
    def _build_device_card_html(
        device: DeviceInfo,
        is_selected: bool = False,
        is_recommended: bool = False,
    ) -> str:
        """Build HTML for a single device card."""
        css_classes = ["crackseg-device-card"]

        if is_selected:
            css_classes.append("selected")
        if is_recommended:
            css_classes.append("recommended")
        if not device.is_available:
            css_classes.append("unavailable")

        # Build badges
        badges = []
        if is_recommended:
            badges.append(
                '<span class="crackseg-device-badge recommended">'
                "Recommended</span>"
            )
        if is_selected:
            badges.append(
                '<span class="crackseg-device-badge selected">Selected</span>'
            )
        if not device.is_available:
            badges.append(
                '<span class="crackseg-device-badge unavailable">'
                "Unavailable</span>"
            )

        badges_html = " ".join(badges)

        # Build device info sections
        info_sections = []

        # Memory info for CUDA devices
        if device.device_type == "cuda" and device.memory_total:
            info_sections.append(
                f"""
                <div class="crackseg-device-info-item">
                    <div class="crackseg-device-info-label">Total Memory</div>
                    <div class="crackseg-device-info-value">
                        {device.memory_total:.1f} GB
                    </div>
                </div>
                """
            )

            if device.memory_available:
                info_sections.append(
                    f"""
                    <div class="crackseg-device-info-item">
                        <div class="crackseg-device-info-label">
                            Available Memory
                        </div>
                        <div class="crackseg-device-info-value">
                            {device.memory_available:.1f} GB
                        </div>
                    </div>
                    """
                )

        # Compute capability for CUDA devices
        if device.compute_capability:
            info_sections.append(
                f"""
                <div class="crackseg-device-info-item">
                    <div class="crackseg-device-info-label">
                        Compute Capability
                    </div>
                    <div class="crackseg-device-info-value">
                        {device.compute_capability}
                    </div>
                </div>
                """
            )

        # Device ID
        info_sections.append(
            f"""
            <div class="crackseg-device-info-item">
                <div class="crackseg-device-info-label">Device ID</div>
                <div class="crackseg-device-info-value">
                    {device.device_id}
                </div>
            </div>
        """
        )

        info_html = "".join(info_sections)

        return f"""
        <div class="{" ".join(css_classes)}"
             data-device-id="{device.device_id}">
            <div class="crackseg-device-header">
                <h3 class="crackseg-device-title">{device.device_name}</h3>
                <span class="crackseg-device-type {device.device_type}">
                    {device.device_type.upper()}
                </span>
            </div>
            <div class="crackseg-device-info">
                {info_html}
            </div>
            <div class="crackseg-device-badges">
                {badges_html}
            </div>
        </div>
        """

    @staticmethod
    @track_performance_decorator("device_selector_render")
    def render_device_selector(
        selected_device: str | None = None,
        component_id: str = "device_selector",
        session_key: str = "selected_device",
        show_title: bool = True,
    ) -> str:
        """Render the device selector component."""
        error_handler = StandardErrorState("DeviceSelector")

        try:
            # Ensure CSS is injected
            OptimizedDeviceSelector._ensure_css_injected()

            # Get available devices
            devices = DeviceDetector.get_available_devices()
            recommended_device = DeviceDetector.get_recommended_device()

            # Build device options for selectbox
            device_options = []
            device_labels = []

            for device in devices:
                device_options.append(device.device_id)

                # Create label with device info
                label = f"{device.device_name} ({device.device_id})"
                if device.device_id == recommended_device:
                    label += " [Recommended]"
                if not device.is_available:
                    label += " [Unavailable]"

                device_labels.append(label)

            # Create device mapping
            device_map = dict(zip(device_labels, device_options, strict=False))

            # Get current selection from Streamlit session state
            current_selection = selected_device
            if current_selection is None:
                # Use Streamlit's session state directly
                current_selection = st.session_state.get(
                    session_key, recommended_device
                )

            # Find current label
            current_label = None
            for label, device_id in device_map.items():
                if device_id == current_selection:
                    current_label = label
                    break

            # Render header if requested
            if show_title:
                header_html = """
                <div class="crackseg-device-selection-container">
                    <h2 class="crackseg-device-selection-title">
                        Device Selection
                    </h2>
                    <p class="crackseg-device-selection-subtitle">
                        Choose the device for training operations. GPU devices
                        provide faster training.
                    </p>
                </div>
                """
                st.markdown(header_html, unsafe_allow_html=True)

            # Render device cards
            cards_html = '<div class="crackseg-device-selector">'

            for device in devices:
                is_selected = device.device_id == current_selection
                is_recommended = device.device_id == recommended_device

                card_html = OptimizedDeviceSelector._build_device_card_html(
                    device, is_selected, is_recommended
                )
                cards_html += card_html

            cards_html += "</div>"

            # Display device cards
            st.markdown(cards_html, unsafe_allow_html=True)

            # Device selection dropdown
            selected_label = st.selectbox(
                "Select Device",
                options=device_labels,
                index=(
                    device_labels.index(current_label) if current_label else 0
                ),
                key=f"{component_id}_selectbox",
                help="Choose the device for training operations",
            )

            # Get selected device ID
            selected_device_id = device_map[selected_label]

            # Update Streamlit session state directly
            st.session_state[session_key] = selected_device_id

            # Clear any previous errors
            error_handler.clear_error()

            return selected_device_id

        except Exception as e:
            # Create proper error info
            error_info = ErrorInfo(
                error_type=ErrorType.UNEXPECTED,
                title="Device Selection Error",
                message=f"Failed to render device selector: {str(e)}",
                technical_info=str(e),
                retry_possible=True,
            )
            error_handler.show_error(error_info)
            logger.error(f"Device selector error: {e}")
            return "cpu"  # Safe fallback

    @staticmethod
    def get_device_from_selection(selection: str) -> torch.device:
        """Convert device selection to PyTorch device."""
        try:
            return get_device(selection)
        except Exception as e:
            # Log error and fallback to CPU
            logger.error(
                f"Error creating device from selection '{selection}': {e}"
            )
            return torch.device("cpu")


def device_selector(
    selected_device: str | None = None,
    component_id: str = "device_selector",
    session_key: str = "selected_device",
    show_title: bool = True,
) -> str:
    """
    Convenience function for device selector component.

    Args:
        selected_device: Currently selected device ID
        component_id: Unique component identifier
        session_key: Session state key for persistence
        show_title: Whether to show the title header

    Returns:
        Selected device ID
    """
    return OptimizedDeviceSelector.render_device_selector(
        selected_device=selected_device,
        component_id=component_id,
        session_key=session_key,
        show_title=show_title,
    )
