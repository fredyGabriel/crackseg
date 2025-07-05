"""Core ResponsiveDevice class and validation logic.

This module provides the ResponsiveDevice class with validation,
configuration management, and core device functionality.
"""

import logging
from dataclasses import dataclass, field

from ..core import (
    DeviceCategory,
    Orientation,
    TouchCapability,
    ViewportDimensions,
)

logger = logging.getLogger(__name__)


@dataclass
class ResponsiveDevice:
    """Configuration for a responsive testing device."""

    name: str
    category: DeviceCategory
    viewport: ViewportDimensions
    touch_capability: TouchCapability = TouchCapability.NONE
    user_agent: str | None = None

    # Testing preferences
    priority: int = 1  # 1=high, 2=medium, 3=low
    enabled: bool = True
    supports_orientation_change: bool = False

    # Performance characteristics
    expected_performance_multiplier: float = 1.0
    network_simulation: str | None = None  # e.g., "slow_3g", "fast_3g"

    # Additional metadata
    description: str = ""
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate and set device defaults after initialization."""
        self._set_category_defaults()
        self._validate_device_configuration()

    def _set_category_defaults(self) -> None:
        """Set defaults based on device category."""
        if self.category == DeviceCategory.MOBILE:
            if self.touch_capability == TouchCapability.NONE:
                self.touch_capability = TouchCapability.ADVANCED
            if not hasattr(self, "_supports_orientation_set"):
                self.supports_orientation_change = True

        elif self.category == DeviceCategory.TABLET:
            if self.touch_capability == TouchCapability.NONE:
                self.touch_capability = TouchCapability.ADVANCED
            if not hasattr(self, "_supports_orientation_set"):
                self.supports_orientation_change = True

        elif self.category == DeviceCategory.DESKTOP:
            if self.touch_capability not in [
                TouchCapability.NONE,
                TouchCapability.BASIC,
            ]:
                # Some desktop devices have touch, but not advanced gestures
                self.touch_capability = TouchCapability.NONE

    def _validate_device_configuration(self) -> None:
        """Validate device configuration for consistency."""
        # Mobile devices should have smaller screens
        if self.category == DeviceCategory.MOBILE:
            if self.viewport.width > 500 or self.viewport.height > 900:
                logger.warning(
                    f"Mobile device {self.name} has unusually large viewport: "
                    f"{self.viewport.width}x{self.viewport.height}"
                )

        # Desktop devices should have larger screens
        elif self.category == DeviceCategory.DESKTOP:
            if self.viewport.width < 1024 or self.viewport.height < 768:
                logger.warning(
                    f"Desktop device {self.name} has small viewport: "
                    f"{self.viewport.width}x{self.viewport.height}"
                )

        # Validate priority range
        if not 1 <= self.priority <= 3:
            logger.warning(
                f"Device {self.name} has unusual priority {self.priority}. "
                "Expected 1-3 (1=high, 2=medium, 3=low)"
            )

        # Validate performance multiplier
        if self.expected_performance_multiplier <= 0:
            raise ValueError(
                f"Performance multiplier must be positive: "
                f"{self.expected_performance_multiplier}"
            )

    def get_orientations(self) -> list[Orientation]:
        """Get supported orientations for this device.

        Returns:
            List of supported orientations
        """
        if not self.supports_orientation_change:
            return [self.viewport.orientation]

        return [Orientation.PORTRAIT, Orientation.LANDSCAPE]

    def get_viewport_for_orientation(
        self, orientation: Orientation
    ) -> ViewportDimensions:
        """Get viewport dimensions for specific orientation.

        Args:
            orientation: Target orientation

        Returns:
            ViewportDimensions for the specified orientation
        """
        if orientation == Orientation.PORTRAIT:
            return self.viewport.get_portrait_dimensions()
        else:
            return self.viewport.get_landscape_dimensions()

    def supports_touch_interaction(self) -> bool:
        """Check if device supports touch interactions.

        Returns:
            True if device has touch capabilities
        """
        return self.touch_capability != TouchCapability.NONE

    def supports_advanced_gestures(self) -> bool:
        """Check if device supports advanced touch gestures.

        Returns:
            True if device supports advanced gestures
        """
        return self.touch_capability == TouchCapability.ADVANCED

    def get_estimated_test_duration_multiplier(self) -> float:
        """Get estimated test duration multiplier for this device.

        Returns:
            Multiplier for test duration estimation
        """
        base_multiplier = self.expected_performance_multiplier

        # Add overhead for orientation testing
        if self.supports_orientation_change:
            base_multiplier *= 1.5

        # Add overhead for touch testing
        if self.supports_touch_interaction():
            base_multiplier *= 1.2

        return base_multiplier

    def to_dict(self) -> dict[str, any]:
        """Convert device to dictionary representation.

        Returns:
            Dictionary with device properties
        """
        return {
            "name": self.name,
            "category": self.category.value,
            "viewport": self.viewport.to_dict(),
            "touch_capability": self.touch_capability.value,
            "user_agent": self.user_agent,
            "priority": self.priority,
            "enabled": self.enabled,
            "supports_orientation_change": self.supports_orientation_change,
            "expected_performance_multiplier": (
                self.expected_performance_multiplier
            ),
            "network_simulation": self.network_simulation,
            "description": self.description,
            "tags": self.tags,
        }

    def __str__(self) -> str:
        """String representation of responsive device."""
        return (
            f"{self.name} ({self.category.value}) - "
            f"{self.viewport} - {self.touch_capability.value} touch"
        )
