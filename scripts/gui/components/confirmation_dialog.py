"""
Confirmation dialog component for the CrackSeg application.

This module provides accessible confirmation dialogs for critical actions to
prevent accidental operations and improve user confidence. Core classes and
factory methods.
"""

import functools
import time
from enum import Enum
from typing import Any

from scripts.gui.utils.performance_optimizer import (
    get_optimizer,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


def track_performance_decorator(operation: str):
    """Decorator to track performance of dialog operations."""

    def decorator(func: Any) -> Any:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Track performance using the optimizer
                component_id = kwargs.get(
                    "component_id", "confirmation_dialog"
                )
                get_optimizer().track_performance(
                    component_id, operation, start_time
                )

        return wrapper

    return decorator


class ConfirmationAction(Enum):
    """Standard confirmation action types."""

    # Training operations
    START_TRAINING = "start_training"
    STOP_TRAINING = "stop_training"
    RESET_TRAINING = "reset_training"

    # Configuration operations
    RESET_CONFIG = "reset_config"
    DELETE_CONFIG = "delete_config"
    RESTORE_CONFIG = "restore_config"

    # Data operations
    CLEAR_EXPERIMENTS = "clear_experiments"
    DELETE_CHECKPOINTS = "delete_checkpoints"
    PURGE_CACHE = "purge_cache"

    # Device operations
    SWITCH_DEVICE = "switch_device"
    RESET_DEVICE = "reset_device"

    # General operations
    LOGOUT = "logout"
    RESET_SESSION = "reset_session"
    FACTORY_RESET = "factory_reset"


class ConfirmationDialog:
    """Information about a confirmation dialog."""

    def __init__(
        self,
        action: ConfirmationAction,
        title: str,
        message: str,
        warning_text: str | None = None,
        confirm_text: str = "Confirm",
        cancel_text: str = "Cancel",
        danger_level: str = "medium",
        requires_typing: bool = False,
        confirmation_phrase: str | None = None,
    ) -> None:
        """Initialize confirmation dialog."""
        self.action = action
        self.title = title
        self.message = message
        self.warning_text = warning_text
        self.confirm_text = confirm_text
        self.cancel_text = cancel_text
        self.danger_level = danger_level  # low, medium, high
        self.requires_typing = requires_typing
        self.confirmation_phrase = confirmation_phrase

        # Validate inputs
        self._validate_dialog_config()

    def _validate_dialog_config(self) -> None:
        """Validate dialog configuration."""
        if not self.title.strip():
            raise ValueError("Dialog title cannot be empty")
        if not self.message.strip():
            raise ValueError("Dialog message cannot be empty")
        if self.danger_level not in ["low", "medium", "high"]:
            raise ValueError("Danger level must be 'low', 'medium', or 'high'")
        if self.requires_typing and not self.confirmation_phrase:
            raise ValueError(
                "Confirmation phrase required when requires_typing is True"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert dialog to dictionary."""
        return {
            "action": self.action.value,
            "title": self.title,
            "message": self.message,
            "warning_text": self.warning_text,
            "confirm_text": self.confirm_text,
            "cancel_text": self.cancel_text,
            "danger_level": self.danger_level,
            "requires_typing": self.requires_typing,
            "confirmation_phrase": self.confirmation_phrase,
        }


class ConfirmationDialogFactory:
    """Factory for creating standard confirmation dialogs."""

    @staticmethod
    def create_training_start_dialog() -> ConfirmationDialog:
        """Create dialog for starting training."""
        return ConfirmationDialog(
            action=ConfirmationAction.START_TRAINING,
            title="Start Training",
            message=(
                "Are you sure you want to start training with the current "
                "configuration?"
            ),
            warning_text=(
                "This will use GPU resources and may take several hours."
            ),
            confirm_text="Start Training",
            cancel_text="Cancel",
            danger_level="medium",
        )

    @staticmethod
    def create_training_stop_dialog() -> ConfirmationDialog:
        """Create dialog for stopping training."""
        return ConfirmationDialog(
            action=ConfirmationAction.STOP_TRAINING,
            title="Stop Training",
            message=(
                "Are you sure you want to stop the current training session?"
            ),
            warning_text=(
                "Progress will be saved, but training will be interrupted."
            ),
            confirm_text="Stop Training",
            cancel_text="Continue",
            danger_level="medium",
        )

    @staticmethod
    def create_config_reset_dialog() -> ConfirmationDialog:
        """Create dialog for resetting configuration."""
        return ConfirmationDialog(
            action=ConfirmationAction.RESET_CONFIG,
            title="Reset Configuration",
            message="This will reset all settings to default values.",
            warning_text="All current configuration changes will be lost.",
            confirm_text="Reset",
            cancel_text="Cancel",
            danger_level="high",
            requires_typing=True,
            confirmation_phrase="RESET CONFIG",
        )

    @staticmethod
    def create_checkpoint_delete_dialog() -> ConfirmationDialog:
        """Create dialog for deleting checkpoints."""
        return ConfirmationDialog(
            action=ConfirmationAction.DELETE_CHECKPOINTS,
            title="Delete Checkpoints",
            message=(
                "This will permanently delete all saved model checkpoints."
            ),
            warning_text="This action cannot be undone.",
            confirm_text="Delete",
            cancel_text="Cancel",
            danger_level="high",
            requires_typing=True,
            confirmation_phrase="DELETE CHECKPOINTS",
        )

    @staticmethod
    def create_device_switch_dialog(
        current_device: str, new_device: str
    ) -> ConfirmationDialog:
        """Create dialog for switching devices during training."""
        return ConfirmationDialog(
            action=ConfirmationAction.SWITCH_DEVICE,
            title="Switch Device",
            message=f"Switch from {current_device} to {new_device}?",
            warning_text=(
                "This will interrupt current training and may cause data loss."
            ),
            confirm_text="Switch",
            cancel_text="Cancel",
            danger_level="high",
        )

    @staticmethod
    def create_factory_reset_dialog() -> ConfirmationDialog:
        """Create dialog for factory reset."""
        return ConfirmationDialog(
            action=ConfirmationAction.FACTORY_RESET,
            title="Factory Reset",
            message=(
                "This will reset the entire application to default settings."
            ),
            warning_text=(
                "All data, configurations, and saved models will be lost."
            ),
            confirm_text="Factory Reset",
            cancel_text="Cancel",
            danger_level="high",
            requires_typing=True,
            confirmation_phrase="FACTORY RESET",
        )

    @staticmethod
    def create_logout_dialog() -> ConfirmationDialog:
        """Create dialog for logout."""
        return ConfirmationDialog(
            action=ConfirmationAction.LOGOUT,
            title="Logout",
            message="Are you sure you want to logout?",
            warning_text="Unsaved changes may be lost.",
            confirm_text="Logout",
            cancel_text="Stay",
            danger_level="low",
        )

    @staticmethod
    def create_clear_experiments_dialog() -> ConfirmationDialog:
        """Create dialog for clearing experiments."""
        return ConfirmationDialog(
            action=ConfirmationAction.CLEAR_EXPERIMENTS,
            title="Clear Experiments",
            message="This will delete all experiment data and results.",
            warning_text="This action cannot be undone.",
            confirm_text="Clear",
            cancel_text="Cancel",
            danger_level="high",
            requires_typing=True,
            confirmation_phrase="CLEAR EXPERIMENTS",
        )

    @staticmethod
    def create_custom_dialog(
        action: ConfirmationAction,
        title: str,
        message: str,
        **kwargs: Any,
    ) -> ConfirmationDialog:
        """Create a custom confirmation dialog."""
        return ConfirmationDialog(
            action=action,
            title=title,
            message=message,
            **kwargs,
        )


# Import rendering functionality from separate module
# This maintains backward compatibility while keeping file size under limits
try:
    from scripts.gui.components.confirmation_renderer import (
        OptimizedConfirmationDialog,  # type: ignore
    )
    from scripts.gui.components.confirmation_utils import (
        ConfirmationDialogHelper,  # type: ignore
        activate_confirmation_dialog,
        confirmation_dialog,
        is_confirmation_dialog_active,
    )
except ImportError:
    # Graceful fallback if renderer modules are not available
    logger.warning("Confirmation dialog renderer modules not available")

    # Minimal fallback implementations
    def confirmation_dialog(*args: Any, **kwargs: Any) -> str | None:
        """Fallback implementation."""
        logger.error("Confirmation dialog renderer not available")
        return None

    def activate_confirmation_dialog(*args: Any, **kwargs: Any) -> None:
        """Fallback implementation."""
        logger.error("Confirmation dialog renderer not available")

    def is_confirmation_dialog_active(*args: Any, **kwargs: Any) -> bool:
        """Fallback implementation."""
        return False

    class OptimizedConfirmationDialog:
        """Fallback implementation."""

        pass

    class ConfirmationDialogHelper:
        """Fallback implementation."""

        pass
