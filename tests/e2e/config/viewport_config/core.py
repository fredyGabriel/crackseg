"""Core viewport configuration types and enums.

This module provides the fundamental types for responsive design testing,
including device categories, orientations, touch capabilities, and viewport
dimensions management.
"""

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class DeviceCategory(Enum):
    """Device categories for responsive testing."""

    MOBILE = "mobile"
    TABLET = "tablet"
    DESKTOP = "desktop"
    ULTRAWIDE = "ultrawide"


class Orientation(Enum):
    """Device orientation for testing."""

    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"


class TouchCapability(Enum):
    """Touch interaction capabilities."""

    NONE = "none"  # Desktop - no touch
    BASIC = "basic"  # Basic touch support
    ADVANCED = "advanced"  # Advanced gestures (pinch, swipe, etc.)


@dataclass
class ViewportDimensions:
    """Viewport dimensions with metadata and orientation handling."""

    width: int
    height: int
    device_pixel_ratio: float = 1.0
    orientation: Orientation = Orientation.PORTRAIT

    def __post_init__(self) -> None:
        """Validate viewport dimensions after initialization."""
        self._validate_dimensions()

    def _validate_dimensions(self) -> None:
        """Validate viewport dimensions are reasonable.

        Raises:
            ValueError: If dimensions are invalid
        """
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"Viewport dimensions must be positive: "
                f"{self.width}x{self.height}"
            )

        if self.width < 320 or self.height < 240:
            raise ValueError(
                f"Minimum viewport size is 320x240, got "
                f"{self.width}x{self.height}"
            )

        if self.device_pixel_ratio <= 0:
            raise ValueError(
                f"Device pixel ratio must be positive: "
                f"{self.device_pixel_ratio}"
            )

    @property
    def aspect_ratio(self) -> float:
        """Calculate aspect ratio of viewport.

        Returns:
            Width to height ratio
        """
        return self.width / self.height

    @property
    def total_pixels(self) -> int:
        """Calculate total pixel count.

        Returns:
            Total pixels in viewport
        """
        return self.width * self.height

    @property
    def is_landscape(self) -> bool:
        """Check if viewport is in landscape orientation.

        Returns:
            True if width > height
        """
        return self.width > self.height

    @property
    def is_portrait(self) -> bool:
        """Check if viewport is in portrait orientation.

        Returns:
            True if height >= width
        """
        return self.height >= self.width

    def get_landscape_dimensions(self) -> "ViewportDimensions":
        """Get landscape version of this viewport.

        Returns:
            ViewportDimensions configured for landscape
        """
        if self.orientation == Orientation.LANDSCAPE:
            return self

        return ViewportDimensions(
            width=max(self.width, self.height),
            height=min(self.width, self.height),
            device_pixel_ratio=self.device_pixel_ratio,
            orientation=Orientation.LANDSCAPE,
        )

    def get_portrait_dimensions(self) -> "ViewportDimensions":
        """Get portrait version of this viewport.

        Returns:
            ViewportDimensions configured for portrait
        """
        if self.orientation == Orientation.PORTRAIT:
            return self

        return ViewportDimensions(
            width=min(self.width, self.height),
            height=max(self.width, self.height),
            device_pixel_ratio=self.device_pixel_ratio,
            orientation=Orientation.PORTRAIT,
        )

    def scale_to_density(self, target_density: float) -> "ViewportDimensions":
        """Scale viewport to different pixel density.

        Args:
            target_density: Target device pixel ratio

        Returns:
            Scaled ViewportDimensions
        """
        if target_density <= 0:
            raise ValueError(
                f"Target density must be positive: {target_density}"
            )

        scale_factor = target_density / self.device_pixel_ratio

        return ViewportDimensions(
            width=int(self.width * scale_factor),
            height=int(self.height * scale_factor),
            device_pixel_ratio=target_density,
            orientation=self.orientation,
        )

    def to_dict(self) -> dict[str, any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with viewport properties
        """
        return {
            "width": self.width,
            "height": self.height,
            "device_pixel_ratio": self.device_pixel_ratio,
            "orientation": self.orientation.value,
            "aspect_ratio": self.aspect_ratio,
            "total_pixels": self.total_pixels,
        }

    def __str__(self) -> str:
        """String representation of viewport dimensions."""
        return (
            f"{self.width}x{self.height} "
            f"(@{self.device_pixel_ratio}x, {self.orientation.value})"
        )


def create_viewport_dimensions(
    width: int,
    height: int,
    pixel_ratio: float = 1.0,
    force_orientation: Orientation | None = None,
) -> ViewportDimensions:
    """Create viewport dimensions with optional orientation forcing.

    Args:
        width: Viewport width
        height: Viewport height
        pixel_ratio: Device pixel ratio
        force_orientation: Force specific orientation

    Returns:
        Configured ViewportDimensions

    Raises:
        ValueError: If dimensions are invalid
    """
    # Determine natural orientation
    natural_orientation = (
        Orientation.LANDSCAPE if width > height else Orientation.PORTRAIT
    )

    # Apply forced orientation if specified
    if force_orientation and force_orientation != natural_orientation:
        if force_orientation == Orientation.LANDSCAPE:
            width, height = max(width, height), min(width, height)
        else:  # PORTRAIT
            width, height = min(width, height), max(width, height)

    final_orientation = force_orientation or natural_orientation

    return ViewportDimensions(
        width=width,
        height=height,
        device_pixel_ratio=pixel_ratio,
        orientation=final_orientation,
    )


def get_common_viewport_sizes() -> dict[str, ViewportDimensions]:
    """Get dictionary of common viewport sizes for testing.

    Returns:
        Dictionary mapping size names to ViewportDimensions
    """
    return {
        # Mobile sizes
        "iphone_se": create_viewport_dimensions(375, 667, 2.0),
        "iphone_12": create_viewport_dimensions(390, 844, 3.0),
        "pixel_5": create_viewport_dimensions(393, 851, 2.75),
        "galaxy_s21": create_viewport_dimensions(360, 800, 3.0),
        # Tablet sizes
        "ipad": create_viewport_dimensions(768, 1024, 2.0),
        "ipad_pro": create_viewport_dimensions(1024, 1366, 2.0),
        "surface_duo": create_viewport_dimensions(540, 720, 2.5),
        # Desktop sizes
        "laptop_small": create_viewport_dimensions(1366, 768, 1.0),
        "desktop_hd": create_viewport_dimensions(1920, 1080, 1.0),
        "desktop_4k": create_viewport_dimensions(3840, 2160, 1.0),
        # Ultrawide
        "ultrawide_2k": create_viewport_dimensions(2560, 1440, 1.0),
        "ultrawide_4k": create_viewport_dimensions(3440, 1440, 1.0),
    }


def validate_viewport_dimensions(
    width: int, height: int, pixel_ratio: float = 1.0
) -> bool:
    """Validate viewport dimensions without creating object.

    Args:
        width: Viewport width
        height: Viewport height
        pixel_ratio: Device pixel ratio

    Returns:
        True if dimensions are valid
    """
    try:
        ViewportDimensions(width, height, pixel_ratio)
        return True
    except ValueError:
        return False
