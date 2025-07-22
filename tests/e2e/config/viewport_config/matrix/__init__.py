"""
Test matrix configuration module for responsive testing. This module
provides ResponsiveTestMatrix class and predefined matrix
configurations for different testing scenarios.
"""

from .core import ResponsiveTestMatrix
from .presets import (
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
    # Core matrix class
    "ResponsiveTestMatrix",
    # Preset matrix configurations
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
