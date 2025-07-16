"""
Confirmation dialog utilities for the CrackSeg application.

This module provides convenience functions and integration helpers for
confirmation dialogs. Separated to maintain file size limits and modularity.
"""

from scripts.gui.components.confirmation_dialog import ConfirmationDialog
from scripts.gui.components.confirmation_renderer import (
    OptimizedConfirmationDialog,
)


def confirmation_dialog(
    dialog: ConfirmationDialog,
    component_id: str = "confirmation_dialog",
    session_key: str = "confirmation_dialog_state",
) -> str | None:
    """
    Convenience function for displaying confirmation dialogs.

    Args:
        dialog: ConfirmationDialog instance
        component_id: Unique component identifier
        session_key: Session state key

    Returns:
        "confirmed", "cancelled", or None if pending
    """
    return OptimizedConfirmationDialog.show_confirmation_dialog(
        dialog=dialog,
        component_id=component_id,
        session_key=session_key,
    )


def activate_confirmation_dialog(
    dialog: ConfirmationDialog,
    session_key: str = "confirmation_dialog_state",
) -> None:
    """Activate a confirmation dialog."""
    OptimizedConfirmationDialog.activate_dialog(dialog, session_key)


def is_confirmation_dialog_active(
    session_key: str = "confirmation_dialog_state",
) -> bool:
    """Check if a confirmation dialog is active."""
    return OptimizedConfirmationDialog.is_dialog_active(session_key)


class ConfirmationDialogHelper:
    """Helper class for managing confirmation dialogs in components."""

    def __init__(self, session_key_prefix: str = "dialog") -> None:
        """Initialize helper with session key prefix."""
        self.session_key_prefix = session_key_prefix

    def get_session_key(self, dialog_type: str) -> str:
        """Get session key for a dialog type."""
        return f"{self.session_key_prefix}_{dialog_type}"

    def show_training_start_confirmation(
        self, component_id: str = "training_start"
    ) -> str | None:
        """Show training start confirmation dialog."""
        from scripts.gui.components.confirmation_dialog import (
            ConfirmationDialogFactory,
        )

        dialog = ConfirmationDialogFactory.create_training_start_dialog()
        session_key = self.get_session_key("training_start")

        return confirmation_dialog(
            dialog=dialog,
            component_id=component_id,
            session_key=session_key,
        )

    def show_training_stop_confirmation(
        self, component_id: str = "training_stop"
    ) -> str | None:
        """Show training stop confirmation dialog."""
        from scripts.gui.components.confirmation_dialog import (
            ConfirmationDialogFactory,
        )

        dialog = ConfirmationDialogFactory.create_training_stop_dialog()
        session_key = self.get_session_key("training_stop")

        return confirmation_dialog(
            dialog=dialog,
            component_id=component_id,
            session_key=session_key,
        )

    def show_config_reset_confirmation(
        self, component_id: str = "config_reset"
    ) -> str | None:
        """Show configuration reset confirmation dialog."""
        from scripts.gui.components.confirmation_dialog import (
            ConfirmationDialogFactory,
        )

        dialog = ConfirmationDialogFactory.create_config_reset_dialog()
        session_key = self.get_session_key("config_reset")

        return confirmation_dialog(
            dialog=dialog,
            component_id=component_id,
            session_key=session_key,
        )

    def show_checkpoint_delete_confirmation(
        self, component_id: str = "checkpoint_delete"
    ) -> str | None:
        """Show checkpoint delete confirmation dialog."""
        from scripts.gui.components.confirmation_dialog import (
            ConfirmationDialogFactory,
        )

        dialog = ConfirmationDialogFactory.create_checkpoint_delete_dialog()
        session_key = self.get_session_key("checkpoint_delete")

        return confirmation_dialog(
            dialog=dialog,
            component_id=component_id,
            session_key=session_key,
        )

    def show_device_switch_confirmation(
        self,
        current_device: str,
        new_device: str,
        component_id: str = "device_switch",
    ) -> str | None:
        """Show device switch confirmation dialog."""
        from scripts.gui.components.confirmation_dialog import (
            ConfirmationDialogFactory,
        )

        dialog = ConfirmationDialogFactory.create_device_switch_dialog(
            current_device, new_device
        )
        session_key = self.get_session_key("device_switch")

        return confirmation_dialog(
            dialog=dialog,
            component_id=component_id,
            session_key=session_key,
        )

    def activate_training_start(self) -> None:
        """Activate training start confirmation."""
        from scripts.gui.components.confirmation_dialog import (
            ConfirmationDialogFactory,
        )

        dialog = ConfirmationDialogFactory.create_training_start_dialog()
        session_key = self.get_session_key("training_start")
        activate_confirmation_dialog(dialog, session_key)

    def activate_training_stop(self) -> None:
        """Activate training stop confirmation."""
        from scripts.gui.components.confirmation_dialog import (
            ConfirmationDialogFactory,
        )

        dialog = ConfirmationDialogFactory.create_training_stop_dialog()
        session_key = self.get_session_key("training_stop")
        activate_confirmation_dialog(dialog, session_key)

    def activate_config_reset(self) -> None:
        """Activate configuration reset confirmation."""
        from scripts.gui.components.confirmation_dialog import (
            ConfirmationDialogFactory,
        )

        dialog = ConfirmationDialogFactory.create_config_reset_dialog()
        session_key = self.get_session_key("config_reset")
        activate_confirmation_dialog(dialog, session_key)

    def activate_checkpoint_delete(self) -> None:
        """Activate checkpoint delete confirmation."""
        from scripts.gui.components.confirmation_dialog import (
            ConfirmationDialogFactory,
        )

        dialog = ConfirmationDialogFactory.create_checkpoint_delete_dialog()
        session_key = self.get_session_key("checkpoint_delete")
        activate_confirmation_dialog(dialog, session_key)

    def is_any_dialog_active(self) -> bool:
        """Check if any dialog managed by this helper is active."""
        dialog_types = [
            "training_start",
            "training_stop",
            "config_reset",
            "checkpoint_delete",
            "device_switch",
        ]

        for dialog_type in dialog_types:
            session_key = self.get_session_key(dialog_type)
            if is_confirmation_dialog_active(session_key):
                return True

        return False

    def clear_all_dialogs(self) -> None:
        """Clear all dialogs managed by this helper."""
        import streamlit as st

        dialog_types = [
            "training_start",
            "training_stop",
            "config_reset",
            "checkpoint_delete",
            "device_switch",
        ]

        for dialog_type in dialog_types:
            session_key = self.get_session_key(dialog_type)
            if session_key in st.session_state:
                st.session_state[session_key]["active"] = False
