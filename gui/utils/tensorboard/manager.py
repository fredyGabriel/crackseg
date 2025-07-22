"""Unified TensorBoard manager interface for backward compatibility.

This module provides a unified interface to all TensorBoard management
components after refactoring the original monolithic tb_manager.py file.
It maintains backward compatibility while exposing the modular architecture.

Components:
- Core types and exceptions (core)
- Port management and allocation (port_management)
- Process management and control (process_manager)
- Lifecycle management and automation (lifecycle_manager)

Example:
    >>> # Direct usage of refactored components
    >>> from gui.utils.tb_manager_refactored import (
    ...     TensorBoardManager, TensorBoardLifecycleManager,
    ...     PortRegistry, TensorBoardState
    ... )
    >>>
    >>> # Or use convenience functions
    >>> manager = create_default_tensorboard_setup()
    >>> lifecycle = initialize_complete_lifecycle()
"""

# Re-export core types and exceptions
from .core import (
    LogDirectoryError,
    PortConflictError,
    ProcessStartupError,
    TensorBoardError,
    TensorBoardInfo,
    TensorBoardState,
    create_tensorboard_url,
    format_uptime,
    validate_log_directory,
    validate_port_number,
)

# Re-export lifecycle management components
from .lifecycle_manager import (
    TensorBoardLifecycleManager,
    cleanup_tensorboard_lifecycle,
    get_global_lifecycle_manager,
    initialize_tensorboard_lifecycle,
)

# Re-export port management components
from .port_management import (
    PortAllocation,
    PortRange,
    PortRegistry,
    discover_available_port,
    is_port_available,
)

# Re-export process management components
from .process_manager import (
    TensorBoardManager,
    create_tensorboard_manager,
    get_default_tensorboard_manager,
)

# Module version for tracking
__version__ = "2.0.0"
__refactored_from__ = "tb_manager.py"

# Backward compatibility aliases
TBManager = TensorBoardManager  # Common abbreviation
TBLifecycle = TensorBoardLifecycleManager  # Common abbreviation
TBState = TensorBoardState  # Common abbreviation
TBError = TensorBoardError  # Common abbreviation


def create_default_tensorboard_setup(
    port_range: tuple[int, int] | None = None,
    preferred_port: int = 6006,
    host: str = "localhost",
    auto_lifecycle: bool = True,
) -> tuple[TensorBoardManager, TensorBoardLifecycleManager | None]:
    """Create a complete TensorBoard setup with manager and lifecycle.

    Convenience function that creates both a TensorBoard manager and
    optional lifecycle manager with sensible defaults.

    Args:
        port_range: Tuple of (start, end) ports for discovery
        preferred_port: Preferred port to try first
        host: Host interface for binding
        auto_lifecycle: Whether to create and configure lifecycle manager

    Returns:
        Tuple of (TensorBoardManager, TensorBoardLifecycleManager or None)

    Example:
        >>> manager, lifecycle = create_default_tensorboard_setup()
        >>> # Manager and lifecycle are ready to use
        >>> success = manager.start_tensorboard(Path("logs"))
    """
    manager = create_tensorboard_manager(
        port_range=port_range,
        preferred_port=preferred_port,
        host=host,
    )

    lifecycle = None
    if auto_lifecycle:
        lifecycle = TensorBoardLifecycleManager(
            manager=manager,
            auto_start_on_training=True,
            auto_stop_on_training_complete=True,
            auto_stop_on_app_exit=True,
        )

    return manager, lifecycle


def initialize_complete_lifecycle(
    manager: TensorBoardManager | None = None,
) -> TensorBoardLifecycleManager:
    """Initialize a complete TensorBoard lifecycle manager.

    Creates a lifecycle manager with full automation enabled and
    registers it as the global instance.

    Args:
        manager: Optional TensorBoard manager (creates default if None)

    Returns:
        Configured TensorBoardLifecycleManager instance

    Example:
        >>> lifecycle = initialize_complete_lifecycle()
        >>> lifecycle.handle_training_state_change("starting", log_dir)
    """
    return TensorBoardLifecycleManager(
        manager=manager or get_default_tensorboard_manager(),
        auto_start_on_training=True,
        auto_stop_on_training_complete=True,
        auto_stop_on_app_exit=True,
        startup_delay=5.0,
    )


def get_port_status_summary() -> dict[str, object]:
    """Get comprehensive port allocation status.

    Returns:
        Dictionary with port registry statistics and allocation details
    """
    stats = PortRegistry.get_registry_stats()
    allocated_ports = PortRegistry.get_allocated_ports()

    return {
        "statistics": stats,
        "allocated_ports": allocated_ports,
        "total_ports_in_use": len(allocated_ports),
        "registry_health": (
            "healthy" if len(allocated_ports) < 10 else "crowded"
        ),
    }


def cleanup_all_tensorboard_resources() -> bool:
    """Clean up all TensorBoard resources and registrations.

    Performs comprehensive cleanup of all TensorBoard managers,
    lifecycle managers, and port registrations.

    Returns:
        True if cleanup was successful, False if errors occurred
    """
    try:
        # Clean up lifecycle management
        cleanup_tensorboard_lifecycle()

        # Stop default manager if running
        default_manager = get_default_tensorboard_manager()
        if default_manager.is_running:
            default_manager.stop_tensorboard()

        # Note: We don't force-clear the port registry as other instances
        # might still be using ports legitimately

        return True
    except Exception:
        return False


# Quick access to commonly used components for convenience
__all__ = [
    # Core types and exceptions
    "TensorBoardError",
    "TensorBoardInfo",
    "TensorBoardState",
    "LogDirectoryError",
    "PortConflictError",
    "ProcessStartupError",
    # Port management
    "PortAllocation",
    "PortRange",
    "PortRegistry",
    # Process management
    "TensorBoardManager",
    # Lifecycle management
    "TensorBoardLifecycleManager",
    # Factory functions
    "create_tensorboard_manager",
    "get_default_tensorboard_manager",
    "initialize_tensorboard_lifecycle",
    "get_global_lifecycle_manager",
    # Convenience functions
    "create_default_tensorboard_setup",
    "initialize_complete_lifecycle",
    "get_port_status_summary",
    "cleanup_all_tensorboard_resources",
    # Utility functions
    "create_tensorboard_url",
    "format_uptime",
    "validate_log_directory",
    "validate_port_number",
    "discover_available_port",
    "is_port_available",
    # Cleanup functions
    "cleanup_tensorboard_lifecycle",
    # Backward compatibility aliases
    "TBManager",
    "TBLifecycle",
    "TBState",
    "TBError",
]
