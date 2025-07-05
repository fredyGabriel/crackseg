"""Core ResponsiveTestMatrix class for test configuration management.

This module provides the ResponsiveTestMatrix class with test configuration
generation, device management, and execution planning functionality.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from ..core import DeviceCategory
from ..devices import ResponsiveDevice, get_default_devices

logger = logging.getLogger(__name__)


@dataclass
class ResponsiveTestMatrix:
    """Matrix of devices and configurations for responsive testing."""

    name: str
    devices: list[ResponsiveDevice] = field(default_factory=list)
    test_orientations: bool = True
    test_touch_interactions: bool = True
    parallel_execution: bool = False
    max_parallel_devices: int = 3

    # Test configuration
    viewport_stabilization_delay: float = 0.5  # seconds
    orientation_change_delay: float = 1.0  # seconds
    touch_simulation_delay: float = 0.2  # seconds

    # Filtering options
    priority_filter: list[int] = field(default_factory=lambda: [1, 2, 3])
    category_filter: list[DeviceCategory] = field(
        default_factory=lambda: [
            DeviceCategory.MOBILE,
            DeviceCategory.TABLET,
            DeviceCategory.DESKTOP,
        ]
    )
    enabled_only: bool = True

    def __post_init__(self) -> None:
        """Initialize matrix with validation."""
        if not self.devices:
            self.devices = list(get_default_devices().values())

        self._validate_matrix_configuration()

    def _validate_matrix_configuration(self) -> None:
        """Validate matrix configuration for consistency."""
        if not self.devices:
            raise ValueError("Test matrix must have at least one device")

        if self.max_parallel_devices < 1:
            raise ValueError("Max parallel devices must be at least 1")

        if self.viewport_stabilization_delay < 0:
            raise ValueError("Viewport stabilization delay cannot be negative")

        # Validate devices are unique by name
        device_names = [device.name for device in self.devices]
        if len(device_names) != len(set(device_names)):
            duplicates = [
                name for name in device_names if device_names.count(name) > 1
            ]
            raise ValueError(f"Duplicate device names found: {duplicates}")

    def get_filtered_devices(self) -> list[ResponsiveDevice]:
        """Get devices filtered by current filter settings.

        Returns:
            List of devices matching filter criteria
        """
        filtered = []

        for device in self.devices:
            # Filter by enabled status
            if self.enabled_only and not device.enabled:
                continue

            # Filter by priority
            if device.priority not in self.priority_filter:
                continue

            # Filter by category
            if device.category not in self.category_filter:
                continue

            filtered.append(device)

        return filtered

    def get_test_configurations(self) -> list[dict[str, Any]]:
        """Generate all test configurations for this matrix.

        Returns:
            List of test configuration dictionaries
        """
        configurations = []
        filtered_devices = self.get_filtered_devices()

        for device in filtered_devices:
            orientations = [device.viewport.orientation]

            if self.test_orientations and device.supports_orientation_change:
                orientations = device.get_orientations()

            for orientation in orientations:
                config = {
                    "device": device,
                    "viewport": device.get_viewport_for_orientation(
                        orientation
                    ),
                    "orientation": orientation,
                    "test_touch": (
                        self.test_touch_interactions
                        and device.supports_touch_interaction()
                    ),
                    "user_agent": device.user_agent,
                    "stabilization_delay": self.viewport_stabilization_delay,
                    "orientation_change_delay": self.orientation_change_delay,
                    "touch_simulation_delay": self.touch_simulation_delay,
                }
                configurations.append(config)

        return configurations

    def get_execution_batches(self) -> list[list[dict[str, Any]]]:
        """Get test configurations organized into execution batches.

        Returns:
            List of batches, each containing configurations for parallel
            execution
        """
        configurations = self.get_test_configurations()

        if not self.parallel_execution or self.max_parallel_devices == 1:
            return [[config] for config in configurations]

        batches = []
        current_batch = []

        for config in configurations:
            current_batch.append(config)

            if len(current_batch) >= self.max_parallel_devices:
                batches.append(current_batch)
                current_batch = []

        # Add remaining configurations
        if current_batch:
            batches.append(current_batch)

        return batches

    def estimate_total_test_duration(
        self, base_test_duration: float = 30.0
    ) -> float:
        """Estimate total test duration for this matrix.

        Args:
            base_test_duration: Base test duration per configuration (seconds)

        Returns:
            Estimated total duration in seconds
        """
        configurations = self.get_test_configurations()

        if not configurations:
            return 0.0

        total_duration = 0.0

        if self.parallel_execution:
            batches = self.get_execution_batches()
            for batch in batches:
                # Duration for batch is max of individual test durations
                batch_duration = max(
                    base_test_duration
                    * config["device"].get_estimated_test_duration_multiplier()
                    for config in batch
                )
                total_duration += batch_duration
        else:
            # Sequential execution
            for config in configurations:
                device_duration = (
                    base_test_duration
                    * config["device"].get_estimated_test_duration_multiplier()
                )
                total_duration += device_duration

        return total_duration

    def get_device_by_name(self, name: str) -> ResponsiveDevice | None:
        """Get device by name from the matrix.

        Args:
            name: Device name to search for

        Returns:
            ResponsiveDevice if found, None otherwise
        """
        for device in self.devices:
            if device.name == name:
                return device
        return None

    def add_device(self, device: ResponsiveDevice) -> None:
        """Add device to the matrix.

        Args:
            device: ResponsiveDevice to add

        Raises:
            ValueError: If device name already exists
        """
        if self.get_device_by_name(device.name):
            raise ValueError(
                f"Device with name '{device.name}' already exists"
            )

        self.devices.append(device)

    def remove_device(self, name: str) -> bool:
        """Remove device from the matrix by name.

        Args:
            name: Name of device to remove

        Returns:
            True if device was removed, False if not found
        """
        for i, device in enumerate(self.devices):
            if device.name == name:
                del self.devices[i]
                return True
        return False

    def to_dict(self) -> dict[str, Any]:
        """Convert matrix to dictionary representation.

        Returns:
            Dictionary with matrix properties
        """
        return {
            "name": self.name,
            "devices": [device.to_dict() for device in self.devices],
            "test_orientations": self.test_orientations,
            "test_touch_interactions": self.test_touch_interactions,
            "parallel_execution": self.parallel_execution,
            "max_parallel_devices": self.max_parallel_devices,
            "viewport_stabilization_delay": self.viewport_stabilization_delay,
            "orientation_change_delay": self.orientation_change_delay,
            "touch_simulation_delay": self.touch_simulation_delay,
            "priority_filter": self.priority_filter,
            "category_filter": [cat.value for cat in self.category_filter],
            "enabled_only": self.enabled_only,
        }

    def __str__(self) -> str:
        """String representation of test matrix."""
        filtered_count = len(self.get_filtered_devices())
        config_count = len(self.get_test_configurations())

        return (
            f"{self.name}: {filtered_count} devices, "
            f"{config_count} configurations"
        )
