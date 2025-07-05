"""Device configuration module for responsive testing.

This module provides ResponsiveDevice class and factory functions for
creating device configurations used in responsive design testing.
"""

from .core import ResponsiveDevice
from .defaults import (
    get_default_devices,
    get_desktop_devices,
    get_high_priority_devices,
    get_mobile_devices,
    get_tablet_devices,
)
from .factories import (
    create_desktop_device,
    create_mobile_device,
    create_tablet_device,
)

__all__ = [
    # Core device class
    "ResponsiveDevice",
    # Factory functions
    "create_mobile_device",
    "create_tablet_device",
    "create_desktop_device",
    # Default device collections
    "get_default_devices",
    "get_mobile_devices",
    "get_tablet_devices",
    "get_desktop_devices",
    "get_high_priority_devices",
]
