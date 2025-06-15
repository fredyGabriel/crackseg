"""Backward compatibility wrapper for TensorBoard management.

This module maintains compatibility with existing code that imports
from the original tb_manager.py location. The actual implementation
has been refactored into the tensorboard/ package.

All new code should import from scripts.gui.utils.tensorboard package:
    from scripts.gui.utils.tensorboard import TensorBoardManager

This compatibility layer will be deprecated in future versions.
"""

import warnings
from pathlib import Path

# Import everything from the new modular structure
from .tensorboard import *  # noqa: F403, F401

# Specific imports for exact compatibility
from .tensorboard import (
    PortAllocation,
    PortRange,
    PortRegistry,
    TensorBoardError,
    TensorBoardInfo,
    TensorBoardLifecycleManager,
    TensorBoardManager,
    TensorBoardState,
    create_default_tensorboard_setup,
    initialize_complete_lifecycle,
)
from .tensorboard.manager import *  # noqa: F403, F401


def __deprecated_warning():
    """Issue deprecation warning for old import style."""
    warnings.warn(
        "Importing from 'tb_manager' is deprecated. "
        "Use 'from scripts.gui.utils.tensorboard import ...' instead. "
        "The tb_manager module will be removed in a future version.",
        DeprecationWarning,
        stacklevel=3,
    )


# Legacy function aliases for exact compatibility
def start_tensorboard_for_training(
    log_dir: str | Path,
    preferred_port: int | None = None,
    auto_lifecycle: bool = True,
) -> tuple[bool, TensorBoardManager, str | None]:
    """Legacy compatibility function.

    Use tensorboard.manager.create_default_tensorboard_setup instead.
    """
    __deprecated_warning()

    if isinstance(log_dir, str):
        log_dir = Path(log_dir)

    manager, lifecycle = create_default_tensorboard_setup(
        preferred_port=preferred_port or 6006,
        auto_lifecycle=auto_lifecycle,
    )

    try:
        success = manager.start_tensorboard(
            log_dir=log_dir, preferred_port=preferred_port, force_restart=True
        )
        url = manager.get_url() if success else None
        return success, manager, url
    except Exception:
        return False, manager, None


def stop_all_tensorboard_instances() -> bool:
    """Legacy compatibility function.

    Use tensorboard.manager.cleanup_all_tensorboard_resources instead.
    """
    __deprecated_warning()
    from .tensorboard.manager import cleanup_all_tensorboard_resources

    return cleanup_all_tensorboard_resources()


def get_tensorboard_status() -> dict[str, object]:
    """Legacy compatibility function.

    Use tensorboard.manager.get_port_status_summary instead.
    """
    __deprecated_warning()
    from .tensorboard.manager import get_port_status_summary

    return get_port_status_summary()


# Legacy module constants
__version__ = "2.0.0-compat"
__legacy_module__ = True

# Issue warning when module is imported
__deprecated_warning()

# Backward compatibility exports
__all__ = [
    # Main classes
    "TensorBoardManager",
    "TensorBoardLifecycleManager",
    "TensorBoardState",
    "TensorBoardInfo",
    "TensorBoardError",
    # Port management
    "PortRegistry",
    "PortAllocation",
    "PortRange",
    # Factory functions
    "create_default_tensorboard_setup",
    "initialize_complete_lifecycle",
    # Legacy functions
    "start_tensorboard_for_training",
    "stop_all_tensorboard_instances",
    "get_tensorboard_status",
]
