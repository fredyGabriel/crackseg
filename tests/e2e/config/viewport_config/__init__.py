"""Viewport configuration for responsive design testing.

This module provides comprehensive viewport and device configuration for
testing responsive design across different screen sizes, device orientations,
and touch capabilities.

Example usage:
    from tests.e2e.config.viewport_config import (
        ResponsiveTestMatrix,
        ResponsiveDevice,
        ViewportDimensions,
        get_comprehensive_matrix,
        get_mobile_focused_matrix,
    )

    # Get a predefined test matrix
    matrix = get_comprehensive_matrix()

    # Create custom device
    custom_device = ResponsiveDevice(
        name="Custom Tablet",
        category=DeviceCategory.TABLET,
        viewport=ViewportDimensions(800, 1200, 2.0),
        touch_capability=TouchCapability.ADVANCED,
    )

    # Add to matrix
    matrix.add_device(custom_device)

    # Get test configurations
    configurations = matrix.get_test_configurations()
"""

from .core import (
    DeviceCategory,
    Orientation,
    TouchCapability,
    ViewportDimensions,
    create_viewport_dimensions,
    get_common_viewport_sizes,
    validate_viewport_dimensions,
)
from .devices import (
    ResponsiveDevice,
    create_desktop_device,
    create_mobile_device,
    create_tablet_device,
    get_default_devices,
    get_desktop_devices,
    get_high_priority_devices,
    get_mobile_devices,
    get_tablet_devices,
)
from .matrix import (
    ResponsiveTestMatrix,
    get_comprehensive_matrix,
    get_cross_category_matrix,
    get_desktop_focused_matrix,
    get_edge_case_matrix,
    get_high_priority_matrix,
    get_mobile_focused_matrix,
    get_performance_test_matrix,
    get_quick_regression_matrix,
    get_smoke_test_matrix,
    get_tablet_focused_matrix,
)

__all__ = [
    # Core types and enums
    "DeviceCategory",
    "Orientation",
    "TouchCapability",
    "ViewportDimensions",
    # Core functions
    "create_viewport_dimensions",
    "get_common_viewport_sizes",
    "validate_viewport_dimensions",
    # Device types and functions
    "ResponsiveDevice",
    "create_mobile_device",
    "create_tablet_device",
    "create_desktop_device",
    "get_default_devices",
    "get_mobile_devices",
    "get_tablet_devices",
    "get_desktop_devices",
    "get_high_priority_devices",
    # Test matrix types and functions
    "ResponsiveTestMatrix",
    "get_smoke_test_matrix",
    "get_comprehensive_matrix",
    "get_mobile_focused_matrix",
    "get_tablet_focused_matrix",
    "get_desktop_focused_matrix",
    "get_high_priority_matrix",
    "get_quick_regression_matrix",
    "get_performance_test_matrix",
    "get_cross_category_matrix",
    "get_edge_case_matrix",
]
