"""
Predefined test matrix configurations for common scenarios. This
module provides factory functions for creating ResponsiveTestMatrix
instances configured for common testing scenarios like smoke tests,
comprehensive testing, and device-specific focused testing.
"""

from ..core import DeviceCategory
from ..devices import get_default_devices
from .core import ResponsiveTestMatrix


def get_smoke_test_matrix() -> ResponsiveTestMatrix:
    """
    Get minimal matrix for smoke testing. Returns: ResponsiveTestMatrix
    configured for quick smoke tests
    """
    devices = get_default_devices()

    return ResponsiveTestMatrix(
        name="Smoke Test",
        devices=[
            devices["iphone_12"],
            devices["desktop_hd"],
        ],
        test_orientations=False,
        test_touch_interactions=False,
        parallel_execution=True,
        max_parallel_devices=2,
        priority_filter=[1],  # High priority only
        viewport_stabilization_delay=0.2,
        orientation_change_delay=0.5,
        touch_simulation_delay=0.1,
    )


def get_comprehensive_matrix() -> ResponsiveTestMatrix:
    """
    Get comprehensive matrix for thorough testing. Returns:
    ResponsiveTestMatrix configured for comprehensive testing
    """
    return ResponsiveTestMatrix(
        name="Comprehensive Test",
        devices=list(get_default_devices().values()),
        test_orientations=True,
        test_touch_interactions=True,
        parallel_execution=True,
        max_parallel_devices=3,
        priority_filter=[1, 2, 3],  # All priorities
    )


def get_mobile_focused_matrix() -> ResponsiveTestMatrix:
    """
    Get matrix focused on mobile devices. Returns: ResponsiveTestMatrix
    configured for mobile-focused testing
    """
    devices = get_default_devices()

    return ResponsiveTestMatrix(
        name="Mobile Focused",
        devices=[
            devices["iphone_12"],
            devices["iphone_se"],
            devices["pixel_5"],
            devices["galaxy_s21"],
            devices["ipad"],
            devices["ipad_pro"],
        ],
        test_orientations=True,
        test_touch_interactions=True,
        parallel_execution=True,
        max_parallel_devices=2,
        category_filter=[DeviceCategory.MOBILE, DeviceCategory.TABLET],
    )


def get_tablet_focused_matrix() -> ResponsiveTestMatrix:
    """
    Get matrix focused on tablet devices. Returns: ResponsiveTestMatrix
    configured for tablet-focused testing
    """
    devices = get_default_devices()

    return ResponsiveTestMatrix(
        name="Tablet Focused",
        devices=[
            devices["ipad"],
            devices["ipad_pro"],
            devices["surface_duo"],
            devices["galaxy_tab"],
        ],
        test_orientations=True,
        test_touch_interactions=True,
        parallel_execution=True,
        max_parallel_devices=3,
        category_filter=[DeviceCategory.TABLET],
    )


def get_desktop_focused_matrix() -> ResponsiveTestMatrix:
    """
    Get matrix focused on desktop devices. Returns: ResponsiveTestMatrix
    configured for desktop-focused testing
    """
    devices = get_default_devices()

    return ResponsiveTestMatrix(
        name="Desktop Focused",
        devices=[
            devices["laptop_small"],
            devices["laptop_standard"],
            devices["desktop_hd"],
            devices["desktop_wide"],
            devices["desktop_4k"],
            devices["ultrawide"],
        ],
        test_orientations=False,  # Desktop doesn't change orientation
        test_touch_interactions=False,  # Most desktops don't have touch
        parallel_execution=True,
        max_parallel_devices=3,
        category_filter=[DeviceCategory.DESKTOP],
    )


def get_high_priority_matrix() -> ResponsiveTestMatrix:
    """
    Get matrix with only high priority devices. Returns:
    ResponsiveTestMatrix configured for high priority devices only
    """
    devices = get_default_devices()

    return ResponsiveTestMatrix(
        name="High Priority",
        devices=[
            devices["iphone_12"],
            devices["pixel_5"],
            devices["desktop_hd"],
        ],
        test_orientations=True,
        test_touch_interactions=True,
        parallel_execution=True,
        max_parallel_devices=3,
        priority_filter=[1],  # High priority only
    )


def get_quick_regression_matrix() -> ResponsiveTestMatrix:
    """
    Get matrix for quick regression testing. Returns: ResponsiveTestMatrix
    configured for fast regression tests
    """
    devices = get_default_devices()

    return ResponsiveTestMatrix(
        name="Quick Regression",
        devices=[
            devices["iphone_12"],
            devices["ipad"],
            devices["desktop_hd"],
        ],
        test_orientations=False,  # Skip orientation for speed
        test_touch_interactions=False,  # Skip touch for speed
        parallel_execution=True,
        max_parallel_devices=3,
        priority_filter=[1, 2],  # High and medium priority
        viewport_stabilization_delay=0.2,  # Faster delays
        orientation_change_delay=0.3,
        touch_simulation_delay=0.1,
    )


def get_performance_test_matrix() -> ResponsiveTestMatrix:
    """
    Get matrix for performance testing across device categories. Returns:
    ResponsiveTestMatrix configured for performance testing
    """
    devices = get_default_devices()

    return ResponsiveTestMatrix(
        name="Performance Test",
        devices=[
            devices["iphone_se"],  # Lower end mobile
            devices["iphone_12"],  # Modern mobile
            devices["ipad_pro"],  # High end tablet
            devices["laptop_small"],  # Lower end desktop
            devices["desktop_4k"],  # High end desktop
        ],
        test_orientations=True,
        test_touch_interactions=True,
        parallel_execution=False,  # Sequential for accurate performance
        # measurement
        max_parallel_devices=1,
        priority_filter=[1, 2, 3],
    )


def get_cross_category_matrix() -> ResponsiveTestMatrix:
    """
    Get matrix with representative devices from each category. Returns:
    ResponsiveTestMatrix with balanced device representation
    """
    devices = get_default_devices()

    return ResponsiveTestMatrix(
        name="Cross Category",
        devices=[
            # Mobile representatives
            devices["iphone_12"],
            devices["pixel_5"],
            # Tablet representatives
            devices["ipad"],
            devices["surface_duo"],
            # Desktop representatives
            devices["laptop_standard"],
            devices["desktop_hd"],
        ],
        test_orientations=True,
        test_touch_interactions=True,
        parallel_execution=True,
        max_parallel_devices=2,
        priority_filter=[1, 2],
    )


def get_edge_case_matrix() -> ResponsiveTestMatrix:
    """
    Get matrix for testing edge cases and unusual devices. Returns:
    ResponsiveTestMatrix with edge case device configurations
    """
    devices = get_default_devices()

    return ResponsiveTestMatrix(
        name="Edge Cases",
        devices=[
            devices["iphone_se"],  # Small mobile
            devices["surface_duo"],  # Dual screen
            devices["ultrawide"],  # Ultrawide desktop
            devices["desktop_4k"],  # High resolution
        ],
        test_orientations=True,
        test_touch_interactions=True,
        parallel_execution=True,
        max_parallel_devices=2,
        priority_filter=[2, 3],  # Focus on less common devices
    )
