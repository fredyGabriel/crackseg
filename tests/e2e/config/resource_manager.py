"""Resource management for parallel test execution.

This module provides resource allocation, port management, and worker isolation
for parallel test execution, ensuring proper resource coordination and
preventing
conflicts between test workers.
"""

import logging
import os
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ResourceAllocation:
    """Resource allocation tracking for a test worker."""

    worker_id: str
    memory_limit_mb: int
    cpu_limit: int
    allocated_ports: list[int] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    environment_vars: dict[str, str] = field(default_factory=dict)

    @property
    def allocation_duration(self) -> float:
        """Get allocation duration in seconds."""
        return time.time() - self.start_time


class PortManager:
    """Manages port allocation for test workers to prevent conflicts."""

    def __init__(self, port_range: tuple[int, int] = (8600, 8699)) -> None:
        """Initialize port manager.

        Args:
            port_range: Tuple of (start_port, end_port) for allocation
        """
        self.start_port, self.end_port = port_range
        self._allocated_ports: set[int] = set()
        self._lock = threading.Lock()
        self.logger = logging.getLogger(f"{__name__}.PortManager")

    def allocate_port(self) -> int:
        """Allocate an available port.

        Returns:
            Port number that has been allocated

        Raises:
            RuntimeError: If no ports are available
        """
        with self._lock:
            for port in range(self.start_port, self.end_port + 1):
                if port not in self._allocated_ports:
                    self._allocated_ports.add(port)
                    self.logger.debug(f"Allocated port {port}")
                    return port

            raise RuntimeError(
                "No available ports in range "
                f"{self.start_port}-{self.end_port}"
            )

    def deallocate_port(self, port: int) -> None:
        """Deallocate a previously allocated port.

        Args:
            port: Port number to deallocate
        """
        with self._lock:
            self._allocated_ports.discard(port)
            self.logger.debug(f"Deallocated port {port}")

    def allocate_multiple_ports(self, count: int) -> list[int]:
        """Allocate multiple ports at once.

        Args:
            count: Number of ports to allocate

        Returns:
            List of allocated port numbers
        """
        ports = []
        try:
            for _ in range(count):
                ports.append(self.allocate_port())
            return ports
        except RuntimeError:
            # Clean up partially allocated ports
            for port in ports:
                self.deallocate_port(port)
            raise

    def get_allocated_ports(self) -> set[int]:
        """Get copy of currently allocated ports."""
        with self._lock:
            return self._allocated_ports.copy()


class MemoryMonitor:
    """Monitor and enforce memory limits for test workers."""

    def __init__(self) -> None:
        """Initialize memory monitor."""
        self.logger = logging.getLogger(f"{__name__}.MemoryMonitor")

    def check_memory_limit(self, limit_mb: int) -> bool:
        """Check if current memory usage is within limit.

        Args:
            limit_mb: Memory limit in megabytes

        Returns:
            True if within limit, False otherwise
        """
        try:
            import psutil

            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)

            if memory_mb > limit_mb:
                self.logger.warning(
                    f"Memory usage {memory_mb:.1f}MB exceeds limit "
                    f"{limit_mb}MB"
                )
                return False

            return True
        except ImportError:
            self.logger.warning("psutil not available for memory monitoring")
            return True
        except Exception as e:
            self.logger.error(f"Error checking memory limit: {e}")
            return True


class WorkerIsolation:
    """Provides process-level isolation for test workers."""

    def __init__(self) -> None:
        """Initialize worker isolation."""
        self.logger = logging.getLogger(f"{__name__}.WorkerIsolation")

    @contextmanager
    def isolate_process(self, worker_id: str) -> Generator[None, None, None]:
        """Context manager for process isolation.

        Args:
            worker_id: Unique identifier for the worker

        Yields:
            None - provides isolated execution context
        """
        self.logger.debug(f"Starting process isolation for worker {worker_id}")

        # Set worker-specific environment variables
        original_env = os.environ.copy()
        os.environ["TEST_WORKER_ID"] = worker_id
        os.environ["TEST_WORKER_ISOLATION"] = "true"

        try:
            yield
        finally:
            # Restore original environment
            os.environ.clear()
            os.environ.update(original_env)
            self.logger.debug(
                f"Ended process isolation for worker {worker_id}"
            )


class ResourceManager:
    """Central resource manager for coordinating test execution resources."""

    def __init__(
        self,
        port_manager: PortManager | None = None,
        memory_monitor: MemoryMonitor | None = None,
        worker_isolation: WorkerIsolation | None = None,
    ) -> None:
        """Initialize resource manager.

        Args:
            port_manager: Optional custom port manager
            memory_monitor: Optional custom memory monitor
            worker_isolation: Optional custom worker isolation
        """
        self.port_manager = port_manager or PortManager()
        self.memory_monitor = memory_monitor or MemoryMonitor()
        self.worker_isolation = worker_isolation or WorkerIsolation()
        self.allocations: dict[str, ResourceAllocation] = {}
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)

    @contextmanager
    def acquire_resources(
        self,
        memory_limit_mb: int,
        cpu_limit: int,
        port_count: int = 2,
        worker_id: str | None = None,
    ) -> Generator[ResourceAllocation, None, None]:
        """Acquire resources for a test worker.

        Args:
            memory_limit_mb: Memory limit in megabytes
            cpu_limit: CPU core limit
            port_count: Number of ports to allocate
            worker_id: Optional worker ID (auto-generated if not provided)

        Yields:
            ResourceAllocation with allocated resources

        Raises:
            RuntimeError: If resources cannot be allocated
        """
        if worker_id is None:
            worker_id = f"worker_{int(time.time() * 1000000) % 1000000}"

        self.logger.info(f"Acquiring resources for worker {worker_id}")

        # Allocate ports
        try:
            allocated_ports = self.port_manager.allocate_multiple_ports(
                port_count
            )
        except RuntimeError as e:
            self.logger.error(f"Failed to allocate ports for {worker_id}: {e}")
            raise

        # Create resource allocation
        allocation = ResourceAllocation(
            worker_id=worker_id,
            memory_limit_mb=memory_limit_mb,
            cpu_limit=cpu_limit,
            allocated_ports=allocated_ports,
            environment_vars={
                "TEST_WORKER_ID": worker_id,
                "TEST_MEMORY_LIMIT_MB": str(memory_limit_mb),
                "TEST_CPU_LIMIT": str(cpu_limit),
                "TEST_ALLOCATED_PORTS": ",".join(map(str, allocated_ports)),
            },
        )

        # Register allocation
        with self._lock:
            self.allocations[worker_id] = allocation

        try:
            with self.worker_isolation.isolate_process(worker_id):
                self.logger.info(
                    f"Resources acquired for {worker_id}: "
                    f"memory={memory_limit_mb}MB, "
                    f"cpu={cpu_limit}, "
                    f"ports={allocated_ports}"
                )
                yield allocation
        finally:
            # Clean up resources
            self._cleanup_allocation(worker_id)

    def _cleanup_allocation(self, worker_id: str) -> None:
        """Clean up resources for a worker.

        Args:
            worker_id: Worker ID to clean up
        """
        with self._lock:
            allocation = self.allocations.pop(worker_id, None)

        if allocation:
            # Deallocate ports
            for port in allocation.allocated_ports:
                self.port_manager.deallocate_port(port)

            duration = allocation.allocation_duration
            self.logger.info(
                f"Cleaned up resources for {worker_id} "
                f"(allocated for {duration:.2f}s)"
            )

    def get_active_allocations(self) -> dict[str, ResourceAllocation]:
        """Get copy of currently active resource allocations."""
        with self._lock:
            return self.allocations.copy()

    def get_resource_usage_summary(self) -> dict[str, Any]:
        """Get summary of current resource usage."""
        with self._lock:
            active_workers = len(self.allocations)
            total_memory = sum(
                alloc.memory_limit_mb for alloc in self.allocations.values()
            )
            total_cpu = sum(
                alloc.cpu_limit for alloc in self.allocations.values()
            )
            allocated_ports = self.port_manager.get_allocated_ports()

            return {
                "active_workers": active_workers,
                "total_memory_mb": total_memory,
                "total_cpu_cores": total_cpu,
                "allocated_ports": sorted(allocated_ports),
                "port_utilization": len(allocated_ports)
                / (
                    self.port_manager.end_port
                    - self.port_manager.start_port
                    + 1
                ),
            }


# Global resource manager instance
global_resource_manager = ResourceManager()


# Utility functions for easy access
def allocate_ports(count: int = 1) -> list[int]:
    """Allocate ports using the global resource manager.

    Args:
        count: Number of ports to allocate

    Returns:
        List of allocated port numbers
    """
    return global_resource_manager.port_manager.allocate_multiple_ports(count)


def deallocate_ports(ports: list[int]) -> None:
    """Deallocate ports using the global resource manager.

    Args:
        ports: List of port numbers to deallocate
    """
    for port in ports:
        global_resource_manager.port_manager.deallocate_port(port)


def get_resource_summary() -> dict[str, Any]:
    """Get resource usage summary from global resource manager.

    Returns:
        Dictionary with resource usage information
    """
    return global_resource_manager.get_resource_usage_summary()


@contextmanager
def acquire_test_resources(
    memory_mb: int = 256, cpu_cores: int = 1, ports: int = 2
) -> Generator[ResourceAllocation, None, None]:
    """Convenience function to acquire test resources.

    Args:
        memory_mb: Memory limit in megabytes
        cpu_cores: Number of CPU cores
        ports: Number of ports to allocate

    Yields:
        ResourceAllocation with allocated resources
    """
    with global_resource_manager.acquire_resources(
        memory_limit_mb=memory_mb, cpu_limit=cpu_cores, port_count=ports
    ) as allocation:
        yield allocation
