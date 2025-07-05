"""Screenshot and video capture module for E2E testing.

This module provides comprehensive capture capabilities for automated testing,
including screenshot capture on failures, video recording of test execution,
and visual regression testing support.

Key Components:
- ScreenshotCapture: Automated screenshot capture and management
- VideoRecording: Test execution video recording capabilities
- VisualRegression: Screenshot comparison and regression testing
- CaptureStorage: File management and retention policies

Integration:
- Mixins for BaseE2ETest integration
- Pytest fixtures for automated capture
- Docker artifact management compatibility
"""

from .screenshot import (
    ScreenshotCapture,
    ScreenshotCaptureMixin,
    ScreenshotConfig,
)
from .storage import (
    CaptureStorage,
    NamingConvention,
    RetentionPolicy,
    StorageConfig,
)
from .video import (
    VideoConfig,
    VideoRecording,
    VideoRecordingMixin,
)
from .visual_regression import (
    ComparisonResult,
    VisualRegression,
    VisualRegressionConfig,
    VisualRegressionMixin,
)

__all__ = [
    # Screenshot capture
    "ScreenshotCapture",
    "ScreenshotConfig",
    "ScreenshotCaptureMixin",
    # Video recording
    "VideoRecording",
    "VideoConfig",
    "VideoRecordingMixin",
    # Visual regression
    "VisualRegression",
    "VisualRegressionConfig",
    "VisualRegressionMixin",
    "ComparisonResult",
    # Storage management
    "CaptureStorage",
    "StorageConfig",
    "RetentionPolicy",
    "NamingConvention",
]
