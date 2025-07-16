"""TensorBoard port management and allocation system.

This module handles dynamic port discovery, allocation tracking, and port
registry management for TensorBoard instances. Designed to prevent port
conflicts and enable multiple TensorBoard processes running concurrently.

Key Features:
- Thread-safe port allocation registry
- Stale allocation cleanup (5-minute timeout)
- Port range configuration and discovery
- Conflict resolution strategies

Example:
    >>> port_range = PortRange(start=6006, end=6020)
    >>> if PortRegistry.allocate_port(6006, "manager_1"):
    ...     print("Port allocated successfully")
"""

import socket
import threading
import time
from dataclasses import dataclass, field
from typing import ClassVar


@dataclass
class PortAllocation:
    """Information about a port allocation.

    Tracks allocation metadata including timing, ownership, and process
    information for proper lifecycle management and conflict resolution.

    Attributes:
        port: Port number that was allocated
        allocated_at: Timestamp when allocation was created
        process_id: Process ID using this port (if known)
        manager_id: Unique identifier of the manager that allocated this port
        reserved: Whether port is reserved vs actively used
        last_verified: Last time the allocation was verified as active
    """

    port: int
    allocated_at: float = field(default_factory=time.time)
    process_id: int | None = None
    manager_id: str | None = None
    reserved: bool = False
    last_verified: float | None = None


@dataclass
class PortRange:
    """Configuration for port discovery range.

    Defines the search space for available ports, with support for
    preferred port selection and range boundaries.

    Attributes:
        start: First port in the range (inclusive)
        end: Last port in the range (inclusive)
        preferred: Preferred port to try first
    """

    start: int = 6006
    end: int = 6020
    preferred: int = 6006

    def __post_init__(self) -> None:
        """Validate port range configuration."""
        if self.start < 1024:
            raise ValueError("Start port must be >= 1024 (non-privileged)")
        if self.end <= self.start:
            raise ValueError("End port must be greater than start port")
        if not (self.start <= self.preferred <= self.end):
            raise ValueError("Preferred port must be within the range")


class PortRegistry:
    """Global port allocation registry for TensorBoard instances.

    Thread-safe registry that tracks port allocations across multiple
    TensorBoard manager instances. Provides automatic cleanup of stale
    allocations and conflict resolution.

    Class Attributes:
        _registry: Global dictionary mapping ports to allocation info
        _lock: Thread lock for safe concurrent access
        STALE_TIMEOUT: Seconds after which allocations are considered stale
    """

    _registry: ClassVar[dict[int, PortAllocation]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()
    STALE_TIMEOUT: ClassVar[int] = 300  # 5 minutes

    @classmethod
    def allocate_port(
        cls,
        port: int,
        manager_id: str,
        process_id: int | None = None,
        reserve: bool = True,
    ) -> bool:
        """Allocate a port in the global registry.

        Args:
            port: Port number to allocate
            manager_id: Unique identifier for the manager
            process_id: Process ID if known
            reserve: Whether to reserve the port

        Returns:
            True if port was successfully allocated, False if already in use

        Raises:
            ValueError: If port is outside valid range (1024-65535)
        """
        if not (1024 <= port <= 65535):
            raise ValueError(
                f"Port {port} is outside valid range (1024-65535)"
            )

        with cls._lock:
            if port in cls._registry:
                existing = cls._registry[port]
                # Check if existing allocation is stale
                if time.time() - existing.allocated_at > cls.STALE_TIMEOUT:
                    cls._registry.pop(port, None)
                else:
                    return False

            cls._registry[port] = PortAllocation(
                port=port,
                process_id=process_id,
                manager_id=manager_id,
                reserved=reserve,
            )
            return True

    @classmethod
    def release_port(cls, port: int, manager_id: str) -> bool:
        """Release a port allocation.

        Args:
            port: Port number to release
            manager_id: Manager that owns the allocation

        Returns:
            True if port was released, False if not found or not owned
        """
        with cls._lock:
            allocation = cls._registry.get(port)
            if allocation and allocation.manager_id == manager_id:
                cls._registry.pop(port, None)
                return True
            return False

    @classmethod
    def is_port_allocated(cls, port: int) -> bool:
        """Check if a port is allocated in the registry.

        Automatically cleans up stale allocations during check.

        Args:
            port: Port number to check

        Returns:
            True if port is currently allocated, False otherwise
        """
        with cls._lock:
            allocation = cls._registry.get(port)
            if allocation:
                # Clean up stale allocations
                if time.time() - allocation.allocated_at > cls.STALE_TIMEOUT:
                    cls._registry.pop(port, None)
                    return False
                return True
            return False

    @classmethod
    def get_allocated_ports(cls) -> list[int]:
        """Get list of currently allocated ports.

        Performs cleanup of stale allocations as part of the operation.

        Returns:
            List of port numbers that are currently allocated
        """
        with cls._lock:
            current_time = time.time()
            # Clean up stale allocations
            stale_ports = [
                port
                for port, allocation in cls._registry.items()
                if current_time - allocation.allocated_at > cls.STALE_TIMEOUT
            ]
            for port in stale_ports:
                cls._registry.pop(port, None)

            return list(cls._registry.keys())

    @classmethod
    def update_process_id(
        cls, port: int, process_id: int, manager_id: str
    ) -> bool:
        """Update the process ID for a port allocation.

        Args:
            port: Port number to update
            process_id: New process ID
            manager_id: Manager that owns the allocation

        Returns:
            True if allocation was updated, False if not found or not owned
        """
        with cls._lock:
            allocation = cls._registry.get(port)
            if allocation and allocation.manager_id == manager_id:
                allocation.process_id = process_id
                allocation.last_verified = time.time()
                return True
            return False

    @classmethod
    def get_allocation_info(cls, port: int) -> PortAllocation | None:
        """Get allocation information for a specific port.

        Args:
            port: Port number to query

        Returns:
            PortAllocation information or None if not allocated
        """
        with cls._lock:
            allocation = cls._registry.get(port)
            if allocation:
                # Check if allocation is stale
                if time.time() - allocation.allocated_at > cls.STALE_TIMEOUT:
                    cls._registry.pop(port, None)
                    return None
                return allocation
            return None

    @classmethod
    def force_release_port(cls, port: int) -> bool:
        """Force release a port (administrative function).

        Args:
            port: Port number to force release

        Returns:
            True if port was released, False if not found
        """
        with cls._lock:
            if port in cls._registry:
                cls._registry.pop(port, None)
                return True
            return False

    @classmethod
    def cleanup_stale_allocations(cls) -> int:
        """Clean up all stale allocations.

        Returns:
            Number of stale allocations that were cleaned up
        """
        with cls._lock:
            current_time = time.time()
            stale_ports = [
                port
                for port, allocation in cls._registry.items()
                if current_time - allocation.allocated_at > cls.STALE_TIMEOUT
            ]

            for port in stale_ports:
                cls._registry.pop(port, None)

            return len(stale_ports)

    @classmethod
    def get_registry_stats(cls) -> dict[str, int]:
        """Get statistics about the port registry.

        Returns:
            Dictionary with registry statistics including total allocations,
            reserved ports, and active processes
        """
        with cls._lock:
            cls.cleanup_stale_allocations()  # Clean up first

            stats = {
                "total_allocations": len(cls._registry),
                "reserved_ports": sum(
                    1 for a in cls._registry.values() if a.reserved
                ),
                "active_processes": sum(
                    1
                    for a in cls._registry.values()
                    if a.process_id is not None
                ),
            }
            return stats


def is_port_available(port: int) -> bool:
    """Check if a port is available for binding.

    Tests both TCP socket binding and registry allocation status.

    Args:
        port: Port number to test

    Returns:
        True if port is available, False if in use or allocated
    """
    # Check registry first
    if PortRegistry.is_port_allocated(port):
        return False

    # Test actual socket binding
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            sock.bind(("localhost", port))
            return True
    except OSError:
        return False


def discover_available_port(
    port_range: PortRange,
    preferred: int | None = None,
) -> int | None:
    """Discover an available port within the specified range.

    Attempts to find ports in the following order:
    1. Preferred port (if specified and in range)
    2. Default preferred port from range
    3. Sequential search from start to end
    4. Return None if no ports available

    Args:
        port_range: Range configuration for port discovery
        preferred: Optional preferred port to try first

    Returns:
        Available port number or None if no ports available
    """
    # Try preferred port first
    if preferred and port_range.start <= preferred <= port_range.end:
        if is_port_available(preferred):
            return preferred

    # Try range preferred port
    if preferred != port_range.preferred and is_port_available(
        port_range.preferred
    ):
        return port_range.preferred

    # Sequential search through range
    for port in range(port_range.start, port_range.end + 1):
        if port != preferred and port != port_range.preferred:
            if is_port_available(port):
                return port

    return None
