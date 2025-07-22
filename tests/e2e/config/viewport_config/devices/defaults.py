"""
Default device configurations for responsive testing. This module
provides predefined device configurations for common mobile, tablet,
and desktop devices used in responsive design testing.
"""

from .core import ResponsiveDevice
from .factories import (
    create_desktop_device,
    create_mobile_device,
    create_tablet_device,
)


def get_default_devices() -> dict[str, ResponsiveDevice]:
    """
    Get dictionary of default device configurations. Returns: Dictionary
    mapping device keys to ResponsiveDevice instances
    """
    return {
        # Mobile devices
        "iphone_12": create_mobile_device(
            name="iPhone 12",
            width=390,
            height=844,
            pixel_ratio=3.0,
            priority=1,
            user_agent=(
                "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 "
                "Mobile/15E148 Safari/604.1"
            ),
            tags=["ios", "mobile", "common"],
        ),
        "iphone_se": create_mobile_device(
            name="iPhone SE",
            width=375,
            height=667,
            pixel_ratio=2.0,
            priority=2,
            user_agent=(
                "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 "
                "Mobile/15E148 Safari/604.1"
            ),
            tags=["ios", "mobile", "small"],
        ),
        "pixel_5": create_mobile_device(
            name="Google Pixel 5",
            width=393,
            height=851,
            pixel_ratio=2.75,
            priority=1,
            user_agent=(
                "Mozilla/5.0 (Linux; Android 11; Pixel 5) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/90.0.4430.91 Mobile Safari/537.36"
            ),
            tags=["android", "mobile", "common"],
        ),
        "galaxy_s21": create_mobile_device(
            name="Samsung Galaxy S21",
            width=360,
            height=800,
            pixel_ratio=3.0,
            priority=2,
            user_agent=(
                "Mozilla/5.0 (Linux; Android 11; SM-G991B) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/90.0.4430.91 Mobile Safari/537.36"
            ),
            tags=["android", "mobile", "samsung"],
        ),
        # Tablet devices
        "ipad": create_tablet_device(
            name="iPad",
            width=768,
            height=1024,
            pixel_ratio=2.0,
            priority=2,
            user_agent=(
                "Mozilla/5.0 (iPad; CPU OS 15_0 like Mac OS X) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 "
                "Mobile/15E148 Safari/604.1"
            ),
            tags=["ios", "tablet", "standard"],
        ),
        "ipad_pro": create_tablet_device(
            name="iPad Pro",
            width=1024,
            height=1366,
            pixel_ratio=2.0,
            priority=2,
            user_agent=(
                "Mozilla/5.0 (iPad; CPU OS 15_0 like Mac OS X) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 "
                "Mobile/15E148 Safari/604.1"
            ),
            tags=["ios", "tablet", "large"],
        ),
        "surface_duo": create_tablet_device(
            name="Microsoft Surface Duo",
            width=540,
            height=720,
            pixel_ratio=2.5,
            priority=3,
            user_agent=(
                "Mozilla/5.0 (Linux; Android 11; Surface Duo) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/90.0.4430.91 Mobile Safari/537.36"
            ),
            tags=["android", "tablet", "dual-screen"],
        ),
        "galaxy_tab": create_tablet_device(
            name="Samsung Galaxy Tab",
            width=800,
            height=1280,
            pixel_ratio=2.0,
            priority=3,
            user_agent=(
                "Mozilla/5.0 (Linux; Android 11; SM-T870) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/90.0.4430.91 Safari/537.36"
            ),
            tags=["android", "tablet", "samsung"],
        ),
        # Desktop viewports
        "laptop_small": create_desktop_device(
            name="Small Laptop",
            width=1366,
            height=768,
            priority=2,
            tags=["laptop", "common", "small"],
        ),
        "laptop_standard": create_desktop_device(
            name="Standard Laptop",
            width=1440,
            height=900,
            priority=2,
            tags=["laptop", "common"],
        ),
        "desktop_hd": create_desktop_device(
            name="Desktop HD",
            width=1920,
            height=1080,
            priority=1,
            tags=["desktop", "common", "hd"],
        ),
        "desktop_wide": create_desktop_device(
            name="Desktop Wide",
            width=2560,
            height=1440,
            priority=2,
            tags=["desktop", "wide", "2k"],
        ),
        "desktop_4k": create_desktop_device(
            name="Desktop 4K",
            width=3840,
            height=2160,
            priority=3,
            expected_performance_multiplier=1.5,
            tags=["desktop", "4k", "high-res"],
        ),
        "ultrawide": create_desktop_device(
            name="Ultrawide Monitor",
            width=3440,
            height=1440,
            priority=3,
            expected_performance_multiplier=1.3,
            tags=["desktop", "ultrawide", "gaming"],
        ),
    }


def get_mobile_devices() -> dict[str, ResponsiveDevice]:
    """
    Get only mobile device configurations. Returns: Dictionary of mobile
    devices
    """
    all_devices = get_default_devices()
    return {
        key: device
        for key, device in all_devices.items()
        if "mobile" in device.tags
    }


def get_tablet_devices() -> dict[str, ResponsiveDevice]:
    """
    Get only tablet device configurations. Returns: Dictionary of tablet
    devices
    """
    all_devices = get_default_devices()
    return {
        key: device
        for key, device in all_devices.items()
        if "tablet" in device.tags
    }


def get_desktop_devices() -> dict[str, ResponsiveDevice]:
    """
    Get only desktop device configurations. Returns: Dictionary of desktop
    devices
    """
    all_devices = get_default_devices()
    return {
        key: device
        for key, device in all_devices.items()
        if "desktop" in device.tags or "laptop" in device.tags
    }


def get_high_priority_devices() -> dict[str, ResponsiveDevice]:
    """
    Get only high priority device configurations. Returns: Dictionary of
    high priority devices (priority = 1)
    """
    all_devices = get_default_devices()
    return {
        key: device
        for key, device in all_devices.items()
        if device.priority == 1
    }
