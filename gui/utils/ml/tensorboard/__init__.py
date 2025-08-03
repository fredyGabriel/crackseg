"""
TensorBoard management components for the GUI interface. This package
provides comprehensive TensorBoard process management, including: -
Core types and exceptions (core) - Port allocation and conflict
resolution (port_management) - Process lifecycle management
(process_manager) - Automated startup/shutdown handling
(lifecycle_manager) - Unified interface for backward compatibility
(manager) Example: >>> from gui.utils.tensorboard import
TensorBoardManager >>> from gui.utils.tensorboard import
TensorBoardLifecycleManager >>> >>> # Or import the unified interface
>>> from gui.utils.tensorboard.manager import ( ...
create_default_tensorboard_setup ... )
"""

# Core exports for easy access
from .core import (
    LogDirectoryError,
    PortConflictError,
    ProcessStartupError,
    TensorBoardError,
    TensorBoardInfo,
    TensorBoardState,
)
from .lifecycle_manager import (
    TensorBoardLifecycleManager,
    cleanup_tensorboard_lifecycle,
    get_global_lifecycle_manager,
    initialize_tensorboard_lifecycle,
)

# Convenience import s from unified manager
from .manager import (
    cleanup_all_tensorboard_resources,
    create_default_tensorboard_setup,
    get_port_status_summary,
    initialize_complete_lifecycle,
)
from .port_management import (
    PortAllocation,
    PortRange,
    PortRegistry,
)
from .process_manager import (
    TensorBoardManager,
    create_tensorboard_manager,
    get_default_tensorboard_manager,
)

__version__ = "2.0.0"
__all__ = [
    # Core components
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
    "create_tensorboard_manager",
    "get_default_tensorboard_manager",
    # Lifecycle management
    "TensorBoardLifecycleManager",
    "cleanup_tensorboard_lifecycle",
    "get_global_lifecycle_manager",
    "initialize_tensorboard_lifecycle",
    # Unified interface
    "create_default_tensorboard_setup",
    "initialize_complete_lifecycle",
    "get_port_status_summary",
    "cleanup_all_tensorboard_resources",
]
