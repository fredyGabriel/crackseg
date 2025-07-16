"""Session state management for TensorBoard component.

This module provides centralized session state management for the
TensorBoard component, handling initialization, updates, and validation.
"""

from pathlib import Path
from typing import Any

import streamlit as st

from ..utils.validators import validate_session_state_keys


class SessionStateManager:
    """Manages session state for TensorBoard component.

    Provides a centralized interface for managing all session state
    related to TensorBoard component operation, including startup
    tracking, error state, and configuration.
    """

    def __init__(self, state_key: str = "tensorboard_component") -> None:
        """Initialize session state manager.

        Args:
            state_key: Unique key for this component's session state.
        """
        self._state_key = state_key
        self._init_session_state()

    def _init_session_state(self) -> None:
        """Initialize component session state with default values."""
        if self._state_key not in st.session_state:
            st.session_state[self._state_key] = {
                "last_log_dir": None,
                "startup_attempted": False,
                "last_status_check": 0,
                "error_message": None,
                "user_initiated_stop": False,
                "startup_start_time": None,
                "startup_progress": 0.0,
                "last_health_check": 0,
                "recovery_attempted": False,
                "error_type": None,
                "startup_attempts": 0,
                "show_advanced_diagnostics": False,
            }

    def get_state(self) -> dict[str, Any]:
        """Get complete component session state.

        Returns:
            Dictionary containing all session state data.

        Raises:
            ValueError: If session state is invalid.
        """
        state = st.session_state[self._state_key]

        # Validate state structure
        is_valid, error_msg = validate_session_state_keys(state)
        if not is_valid:
            raise ValueError(f"Invalid session state: {error_msg}")

        return state

    def update_state(self, **kwargs: Any) -> None:
        """Update session state with new values.

        Args:
            **kwargs: Key-value pairs to update in session state.

        Example:
            >>> manager.update_state(startup_attempted=True,
            ...                      error_message=None)
        """
        current_state = st.session_state[self._state_key]
        current_state.update(kwargs)

        # Validate updated state
        is_valid, error_msg = validate_session_state_keys(current_state)
        if not is_valid:
            # Revert changes if validation fails
            for key in kwargs:
                if key in current_state:
                    del current_state[key]
            raise ValueError(f"Invalid state update: {error_msg}")

    def get_value(self, key: str, default: Any = None) -> Any:
        """Get specific value from session state.

        Args:
            key: State key to retrieve.
            default: Default value if key doesn't exist.

        Returns:
            Value from session state or default.
        """
        return st.session_state[self._state_key].get(key, default)

    def set_value(self, key: str, value: Any) -> None:
        """Set specific value in session state.

        Args:
            key: State key to set.
            value: Value to set.
        """
        self.update_state(**{key: value})

    def reset_error_state(self) -> None:
        """Reset all error-related state to default values."""
        self.update_state(
            error_message=None,
            error_type=None,
            recovery_attempted=False,
            startup_attempts=0,
        )

    def reset_startup_state(self) -> None:
        """Reset all startup-related state to default values."""
        self.update_state(
            startup_attempted=False,
            startup_start_time=None,
            startup_progress=0.0,
            startup_attempts=0,
            recovery_attempted=False,
            error_message=None,
            error_type=None,
            show_advanced_diagnostics=False,
        )

    def is_startup_in_progress(self) -> bool:
        """Check if startup is currently in progress.

        Returns:
            True if startup is in progress, False otherwise.
        """
        return (
            self.get_value("startup_start_time") is not None
            and self.get_value("startup_progress", 0.0) < 1.0
        )

    def has_error(self) -> bool:
        """Check if component is in error state.

        Returns:
            True if there's an active error, False otherwise.
        """
        return bool(self.get_value("error_message"))

    def should_attempt_startup(self, log_dir: Path) -> bool:
        """Determine if automatic startup should be attempted.

        Args:
            log_dir: Current log directory.

        Returns:
            True if startup should be attempted, False otherwise.
        """
        return (
            not self.get_value("startup_attempted", False)
            and not self.get_value("user_initiated_stop", False)
            and self.get_value("startup_attempts", 0) < 3
            and log_dir != self.get_value("last_log_dir")
        )

    def increment_startup_attempts(self) -> int:
        """Increment and return the startup attempt counter.

        Returns:
            New startup attempts count.
        """
        current_attempts = self.get_value("startup_attempts", 0)
        new_attempts = current_attempts + 1
        self.set_value("startup_attempts", new_attempts)
        return new_attempts

    def set_log_directory(self, log_dir: Path) -> None:
        """Update the last log directory and reset related state.

        Args:
            log_dir: New log directory path.
        """
        if log_dir != self.get_value("last_log_dir"):
            self.update_state(
                last_log_dir=log_dir,
                startup_attempted=False,
                startup_attempts=0,
                recovery_attempted=False,
                error_message=None,
                error_type=None,
            )

    def set_error(
        self, error_message: str, error_type: str | None = None
    ) -> None:
        """Set error state with message and optional type.

        Args:
            error_message: Error message to display.
            error_type: Optional error type for categorization.
        """
        self.update_state(
            error_message=error_message,
            error_type=error_type,
            startup_attempted=True,
        )

    def clear_state(self) -> None:
        """Clear all session state for this component."""
        if self._state_key in st.session_state:
            del st.session_state[self._state_key]
        self._init_session_state()

    def export_state(self) -> dict[str, Any]:
        """Export current state for debugging or persistence.

        Returns:
            Copy of current session state.
        """
        return self.get_state().copy()

    def import_state(self, state_data: dict[str, Any]) -> None:
        """Import state data (for testing or restoration).

        Args:
            state_data: State dictionary to import.

        Raises:
            ValueError: If imported state is invalid.
        """
        # Validate imported state
        is_valid, error_msg = validate_session_state_keys(state_data)
        if not is_valid:
            raise ValueError(f"Invalid imported state: {error_msg}")

        # Replace current state
        st.session_state[self._state_key] = state_data.copy()
