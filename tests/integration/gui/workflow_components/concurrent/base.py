"""Base concurrent operation testing infrastructure.

Core components for concurrent operation testing including the main mixin
class and protocol definitions. Split from oversized concurrent_operation_mixin.py.
"""

import threading
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Protocol

from scripts.gui.utils.threading.coordinator import ThreadCoordinator

from .data_integrity import DataIntegrityMixin
from .multi_user import MultiUserMixin
from .resource_contention import ResourceContentionMixin
from .stability import SystemStabilityMixin
from .synchronization import ProcessSynchronizationMixin


class ConcurrentOperationTestUtilities(Protocol):
    """Protocol for test utilities needed by concurrent operation components."""

    temp_path: Path


class ConcurrentOperationMixin(
    MultiUserMixin,
    ResourceContentionMixin,
    ProcessSynchronizationMixin,
    SystemStabilityMixin,
    DataIntegrityMixin,
):
    """Main mixin providing comprehensive concurrent operation testing patterns.

    Combines all concurrent operation testing capabilities into a single
    interface while maintaining modular implementation through multiple mixins.
    """

    def __init__(self) -> None:
        """Initialize concurrent operation testing utilities."""
        self._thread_coordinator = ThreadCoordinator(
            max_workers=8, enable_monitoring=True
        )
        self._active_operations: dict[str, Future[Any]] = {}
        self._operation_results: dict[str, Any] = {}
        self._resource_locks: dict[str, threading.Lock] = {}
        self._operation_lock = threading.Lock()

        # Initialize all mixins
        MultiUserMixin.__init__(self)
        ResourceContentionMixin.__init__(self)
        ProcessSynchronizationMixin.__init__(self)
        SystemStabilityMixin.__init__(self)
        DataIntegrityMixin.__init__(self)

    def cleanup_concurrent_operations(self) -> None:
        """Clean up all concurrent operation resources."""
        # Cancel any active operations
        with self._operation_lock:
            for _name, future in self._active_operations.items():
                future.cancel()
            self._active_operations.clear()

        # Clear results
        self._operation_results.clear()

        # Shutdown thread coordinator
        self._thread_coordinator.shutdown(timeout=5.0)

    def __del__(self) -> None:
        """Ensure cleanup on object destruction."""
        try:
            self.cleanup_concurrent_operations()
        except Exception:
            pass  # Best effort cleanup
