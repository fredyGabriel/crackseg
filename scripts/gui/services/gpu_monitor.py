"""GPU monitoring service for CrackSeg GUI.

This module provides GPU monitoring capabilities for the training interface,
including memory usage tracking and utilization metrics.
"""

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


class GPUMonitor:
    """Monitor GPU status and resources for training operations."""

    def __init__(self) -> None:
        """Initialize GPU monitor."""
        self._memory_history: list[float] = []
        self._utilization_history: list[float] = []
        self._max_history_length = 100

    def get_gpu_info(self) -> dict[str, Any] | None:
        """Get current GPU information.

        Returns:
            Dictionary containing GPU information, or None if no GPU available.
        """
        if not torch.cuda.is_available():
            return None

        try:
            device = torch.cuda.current_device()
            memory_total = torch.cuda.get_device_properties(
                device
            ).total_memory / (1024**3)
            memory_allocated = torch.cuda.memory_allocated(device) / (1024**3)
            memory_reserved = torch.cuda.memory_reserved(device) / (1024**3)

            # Calculate utilization (simplified)
            utilization = (memory_allocated / memory_total) * 100

            # Update history
            self._memory_history.append(memory_allocated)
            self._utilization_history.append(utilization)

            # Limit history length
            if len(self._memory_history) > self._max_history_length:
                self._memory_history.pop(0)
                self._utilization_history.pop(0)

            return {
                "memory_total": memory_total,
                "memory_used": memory_allocated,
                "memory_reserved": memory_reserved,
                "utilization": utilization,
                "memory_history": self._memory_history.copy(),
                "utilization_history": self._utilization_history.copy(),
            }

        except Exception as e:
            logger.error(f"Error getting GPU info: {e}")
            return None

    def clear_history(self) -> None:
        """Clear monitoring history."""
        self._memory_history.clear()
        self._utilization_history.clear()
