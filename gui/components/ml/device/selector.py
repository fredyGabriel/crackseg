"""
Device selector component for the CrackSeg application.

This module provides a high-performance device selector with GPU/CPU detection,
detailed device information, and integration with the existing optimization
system. Allows users to choose between available devices for training
operations.

This is the refactored version that uses modular components for improved
maintainability and adherence to project size limits.
"""

import torch

from crackseg.utils.core.device import get_device
from crackseg.utils.logging import get_logger
from gui.components.device_detector import DeviceDetector
from gui.components.device_info import DeviceInfo
from gui.components.device_selector_ui import (
    DeviceCardRenderer,
    DeviceSelectorCSS,
    DeviceSelectorRenderer,
)

logger = get_logger(__name__)


class OptimizedDeviceSelector:
    """High-performance device selector with caching and optimization.

    This class provides a simplified interface that delegates to the
    modular components for actual implementation.
    """

    # Backward compatibility attributes for existing tests
    _CSS_CONTENT = DeviceSelectorCSS.CONTENT
    _BRAND_COLORS = DeviceSelectorCSS.BRAND_COLORS

    @staticmethod
    def render_device_selector(
        selected_device: str | None = None,
        component_id: str = "device_selector",
        session_key: str = "selected_device",
        show_title: bool = True,
    ) -> str:
        """Render the device selector component.

        Args:
            selected_device: Currently selected device ID
            component_id: Unique component identifier
            session_key: Session state key for persistence
            show_title: Whether to show the title header

        Returns:
            Selected device ID string
        """
        # Call backward compatibility method for tests
        OptimizedDeviceSelector._ensure_css_injected()

        return DeviceSelectorRenderer.render_device_selector(
            selected_device=selected_device,
            component_id=component_id,
            session_key=session_key,
            show_title=show_title,
        )

    @staticmethod
    def get_device_from_selection(selection: str) -> torch.device:
        """Convert device selection to PyTorch device.

        Args:
            selection: Device ID string (e.g., 'cuda:0', 'cpu')

        Returns:
            PyTorch device object
        """
        try:
            return get_device(selection)
        except Exception as e:
            # Log error and fallback to CPU
            logger.error(
                f"Error creating device from selection '{selection}': {e}"
            )
            return torch.device("cpu")

    @staticmethod
    def get_available_devices() -> list[DeviceInfo]:
        """Get list of available devices.

        Convenience method that delegates to DeviceDetector.

        Returns:
            List of DeviceInfo objects
        """
        return DeviceDetector.get_available_devices()

    @staticmethod
    def get_recommended_device() -> str:
        """Get recommended device based on availability and performance.

        Convenience method that delegates to DeviceDetector.

        Returns:
            Device ID string for recommended device
        """
        return DeviceDetector.get_recommended_device()

    # Backward compatibility methods for existing tests
    @staticmethod
    def _build_device_card_html(
        device: DeviceInfo,
        is_selected: bool = False,
        is_recommended: bool = False,
    ) -> str:
        """Build HTML for a device card.

        Backward compatibility method that delegates to DeviceCardRenderer.

        Args:
            device: Device information to display
            is_selected: Whether this device is currently selected
            is_recommended: Whether this device is recommended

        Returns:
            HTML string for the device card
        """
        return DeviceCardRenderer.build_device_card_html(
            device, is_selected, is_recommended
        )

    @staticmethod
    def _ensure_css_injected() -> None:
        """Ensure CSS is injected.

        Backward compatibility method that delegates to DeviceSelectorCSS.
        """
        DeviceSelectorCSS.ensure_css_injected()


def device_selector(
    selected_device: str | None = None,
    component_id: str = "device_selector",
    session_key: str = "selected_device",
    show_title: bool = True,
) -> str:
    """
    Convenience function for device selector component.

    This is the main entry point for the device selector functionality.
    It provides a simple interface that maintains compatibility with
    existing code while using the refactored modular implementation.

    Args:
        selected_device: Currently selected device ID
        component_id: Unique component identifier
        session_key: Session state key for persistence
        show_title: Whether to show the title header

    Returns:
        Selected device ID string

    Example:
        >>> selected = device_selector(show_title=False)
        >>> device = OptimizedDeviceSelector.get_device_from_selection(
        ...     selected
        ... )
        >>> print(f"Using device: {device}")
    """
    return OptimizedDeviceSelector.render_device_selector(
        selected_device=selected_device,
        component_id=component_id,
        session_key=session_key,
        show_title=show_title,
    )


# Convenience functions for common operations
def get_available_devices() -> list[DeviceInfo]:
    """Get list of available devices.

    Returns:
        List of DeviceInfo objects representing available devices
    """
    return DeviceDetector.get_available_devices()


def get_recommended_device() -> str:
    """Get recommended device based on availability and performance.

    Returns:
        Device ID string for the recommended device
    """
    return DeviceDetector.get_recommended_device()


def get_device_from_selection(selection: str) -> torch.device:
    """Convert device selection to PyTorch device.

    Args:
        selection: Device ID string (e.g., 'cuda:0', 'cpu')

    Returns:
        PyTorch device object
    """
    return OptimizedDeviceSelector.get_device_from_selection(selection)
