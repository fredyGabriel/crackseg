"""
Device information data structures for the CrackSeg application.

This module provides basic data structures for representing device information
including DeviceInfo class and related utilities.
"""

from typing import Any


class DeviceInfo:
    """Information about a compute device."""

    def __init__(
        self,
        device_id: str,
        device_type: str,
        device_name: str,
        memory_total: float | None = None,
        memory_available: float | None = None,
        compute_capability: str | None = None,
        is_available: bool = True,
    ) -> None:
        """Initialize device information.

        Args:
            device_id: Unique identifier for the device (e.g., 'cuda:0', 'cpu')
            device_type: Type of device ('cuda', 'cpu', etc.)
            device_name: Human-readable device name
            memory_total: Total memory in GB (if applicable)
            memory_available: Available memory in GB (if applicable)
            compute_capability: CUDA compute capability (if applicable)
            is_available: Whether device is currently accessible
        """
        self.device_id = device_id
        self.device_type = device_type
        self.device_name = device_name
        self.memory_total = memory_total
        self.memory_available = memory_available
        self.compute_capability = compute_capability
        self.is_available = is_available

    def to_dict(self) -> dict[str, Any]:
        """Convert device info to dictionary.

        Returns:
            Dictionary representation of device information
        """
        return {
            "device_id": self.device_id,
            "device_type": self.device_type,
            "device_name": self.device_name,
            "memory_total": self.memory_total,
            "memory_available": self.memory_available,
            "compute_capability": self.compute_capability,
            "is_available": self.is_available,
        }

    def __str__(self) -> str:
        """String representation of device info."""
        return f"DeviceInfo({self.device_id}, {self.device_name})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"DeviceInfo(device_id='{self.device_id}', "
            f"device_type='{self.device_type}', "
            f"device_name='{self.device_name}', "
            f"is_available={self.is_available})"
        )
