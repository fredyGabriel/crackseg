"""
Device detection utilities for the CrackSeg application.

This module provides functionality to detect available computing devices
including CUDA GPUs and CPU, with detailed hardware information.
"""

import torch

from crackseg.utils.logging import get_logger
from scripts.gui.components.device_info import DeviceInfo

logger = get_logger(__name__)


class DeviceDetector:
    """Utility class for detecting available devices."""

    @staticmethod
    def get_available_devices() -> list[DeviceInfo]:
        """Get list of available devices.

        Detects all available computing devices including CPU and CUDA GPUs,
        gathering detailed information about memory, compute capability, etc.

        Returns:
            List of DeviceInfo objects representing available devices
        """
        devices = []

        # Add CPU device
        devices.append(DeviceInfo("cpu", "cpu", "CPU", is_available=True))

        # Add CUDA devices if available
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    device_name = torch.cuda.get_device_name(i)
                    props = torch.cuda.get_device_properties(i)

                    # Memory calculations
                    memory_total = props.total_memory / (1024**3)  # GB
                    memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                    memory_available = memory_total - memory_reserved

                    # Compute capability
                    compute_capability = f"{props.major}.{props.minor}"

                    devices.append(
                        DeviceInfo(
                            f"cuda:{i}",
                            "cuda",
                            device_name,
                            memory_total=memory_total,
                            memory_available=memory_available,
                            compute_capability=compute_capability,
                            is_available=True,
                        )
                    )
                except Exception as e:
                    # Device exists but might not be accessible
                    logger.warning(
                        f"Failed to get info for CUDA device {i}: {e}"
                    )
                    devices.append(
                        DeviceInfo(
                            f"cuda:{i}",
                            "cuda",
                            f"GPU {i}",
                            is_available=False,
                        )
                    )

        return devices

    @staticmethod
    def get_recommended_device() -> str:
        """Get recommended device based on availability and performance.

        Returns the device ID for the recommended device. Prioritizes CUDA
        devices with the most available memory, falling back to CPU if no
        CUDA devices are available.

        Returns:
            Device ID string (e.g., 'cuda:0', 'cpu')
        """
        devices = DeviceDetector.get_available_devices()

        # Prefer CUDA devices with most memory
        cuda_devices = [
            d for d in devices if d.device_type == "cuda" and d.is_available
        ]
        if cuda_devices:
            # Sort by memory total (descending)
            cuda_devices.sort(
                key=lambda d: d.memory_total if d.memory_total else 0,
                reverse=True,
            )
            logger.info(f"Recommended device: {cuda_devices[0].device_id}")
            return cuda_devices[0].device_id

        # Fallback to CPU
        logger.info("Recommended device: CPU (no CUDA devices available)")
        return "cpu"

    @staticmethod
    def get_device_info(device_id: str) -> DeviceInfo | None:
        """Get information for a specific device.

        Args:
            device_id: Device identifier (e.g., 'cuda:0', 'cpu')

        Returns:
            DeviceInfo object if device exists, None otherwise
        """
        devices = DeviceDetector.get_available_devices()
        for device in devices:
            if device.device_id == device_id:
                return device
        return None

    @staticmethod
    def is_device_available(device_id: str) -> bool:
        """Check if a specific device is available.

        Args:
            device_id: Device identifier (e.g., 'cuda:0', 'cpu')

        Returns:
            True if device is available, False otherwise
        """
        device_info = DeviceDetector.get_device_info(device_id)
        return device_info is not None and device_info.is_available
