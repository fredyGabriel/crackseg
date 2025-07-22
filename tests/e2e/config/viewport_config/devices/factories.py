"""
Device factory functions for creating responsive devices. This module
provides factory functions to easily create ResponsiveDevice instances
for different device categories with appropriate defaults.
"""

from ..core import (
    DeviceCategory,
    Orientation,
    TouchCapability,
    ViewportDimensions,
)
from .core import ResponsiveDevice


def create_mobile_device(
    name: str,
    width: int,
    height: int,
    pixel_ratio: float = 2.0,
    user_agent: str | None = None,
    priority: int = 1,
    **kwargs,
) -> ResponsiveDevice:
    """
    Create a mobile device configuration. Args: name: Device name width:
    Viewport width height: Viewport height pixel_ratio: Device pixel ratio
    user_agent: Mobile user agent string priority: Device priority (1-3)
    **kwargs: Additional ResponsiveDevice parameters Returns: Configured
    ResponsiveDevice for mobile
    """
    viewport = ViewportDimensions(
        width, height, pixel_ratio, Orientation.PORTRAIT
    )

    return ResponsiveDevice(
        name=name,
        category=DeviceCategory.MOBILE,
        viewport=viewport,
        touch_capability=TouchCapability.ADVANCED,
        user_agent=user_agent,
        priority=priority,
        supports_orientation_change=True,
        **kwargs,
    )


def create_tablet_device(
    name: str,
    width: int,
    height: int,
    pixel_ratio: float = 2.0,
    user_agent: str | None = None,
    priority: int = 2,
    **kwargs,
) -> ResponsiveDevice:
    """
    Create a tablet device configuration. Args: name: Device name width:
    Viewport width height: Viewport height pixel_ratio: Device pixel ratio
    user_agent: Tablet user agent string priority: Device priority (1-3)
    **kwargs: Additional ResponsiveDevice parameters Returns: Configured
    ResponsiveDevice for tablet
    """
    viewport = ViewportDimensions(
        width, height, pixel_ratio, Orientation.PORTRAIT
    )

    return ResponsiveDevice(
        name=name,
        category=DeviceCategory.TABLET,
        viewport=viewport,
        touch_capability=TouchCapability.ADVANCED,
        user_agent=user_agent,
        priority=priority,
        supports_orientation_change=True,
        **kwargs,
    )


def create_desktop_device(
    name: str,
    width: int,
    height: int,
    pixel_ratio: float = 1.0,
    touch_capability: TouchCapability = TouchCapability.NONE,
    priority: int = 1,
    **kwargs,
) -> ResponsiveDevice:
    """
    Create a desktop device configuration. Args: name: Device name width:
    Viewport width height: Viewport height pixel_ratio: Device pixel ratio
    touch_capability: Touch capability level priority: Device priority
    (1-3) **kwargs: Additional ResponsiveDevice parameters Returns:
    Configured ResponsiveDevice for desktop
    """
    viewport = ViewportDimensions(
        width, height, pixel_ratio, Orientation.LANDSCAPE
    )

    return ResponsiveDevice(
        name=name,
        category=DeviceCategory.DESKTOP,
        viewport=viewport,
        touch_capability=touch_capability,
        priority=priority,
        supports_orientation_change=False,
        **kwargs,
    )
