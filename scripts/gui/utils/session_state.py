"""
Session state management for the CrackSeg application.

This module provides structured session state management with
serialization, validation, and backward compatibility.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st


@dataclass
class SessionState:
    """Structured session state management for the CrackSeg application."""

    # Core application state
    config_path: str | None = None
    run_directory: str | None = None
    current_page: str = "Config"
    theme: str = "dark"

    # Configuration state
    config_loaded: bool = False
    config_data: dict[str, Any] | None = None

    # Training state
    training_active: bool = False
    training_progress: float = 0.0
    training_metrics: dict[str, float] = field(default_factory=dict)

    # Model state
    model_loaded: bool = False
    model_architecture: str | None = None
    model_parameters: dict[str, Any] | None = None

    # Results state
    last_evaluation: dict[str, Any] | None = None
    results_available: bool = False

    # UI state
    sidebar_expanded: bool = True
    notifications: list[str] = field(default_factory=list)

    # Session metadata
    session_id: str = field(
        default_factory=lambda: str(datetime.now().timestamp())
    )
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert session state to dictionary for serialization."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, Path):
                result[key] = str(value)
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionState":
        """Create SessionState from dictionary."""
        # Handle datetime conversion
        if "last_updated" in data and isinstance(data["last_updated"], str):
            data["last_updated"] = datetime.fromisoformat(data["last_updated"])

        return cls(
            **{k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        )

    def update_config(
        self, config_path: str, config_data: dict[str, Any] | None = None
    ) -> None:
        """Update configuration state."""
        self.config_path = config_path
        self.config_data = config_data
        self.config_loaded = config_data is not None
        self.last_updated = datetime.now()

    def update_training_progress(
        self, progress: float, metrics: dict[str, float] | None = None
    ) -> None:
        """Update training progress and metrics."""
        self.training_progress = max(0.0, min(1.0, progress))
        if metrics:
            self.training_metrics.update(metrics)
        self.last_updated = datetime.now()

    def set_training_active(self, active: bool) -> None:
        """Set training active state."""
        self.training_active = active
        if not active:
            self.training_progress = 0.0
        self.last_updated = datetime.now()

    def add_notification(self, message: str) -> None:
        """Add a notification message."""
        self.notifications.append(
            f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
        )
        # Keep only last 10 notifications
        self.notifications = self.notifications[-10:]
        self.last_updated = datetime.now()

    def clear_notifications(self) -> None:
        """Clear all notifications."""
        self.notifications.clear()
        self.last_updated = datetime.now()

    def is_ready_for_training(self) -> bool:
        """Check if application is ready for training."""
        return self.config_loaded and self.run_directory is not None

    def is_ready_for_results(self) -> bool:
        """Check if results can be displayed."""
        return self.results_available or self.run_directory is not None

    def set_theme(self, theme: str) -> None:
        """Update theme setting."""
        self.theme = theme
        self.last_updated = datetime.now()

    def validate(self) -> list[str]:
        """Validate session state and return list of issues."""
        issues = []

        if self.config_path and not Path(self.config_path).exists():
            issues.append("Configuration file does not exist")

        if self.run_directory and not Path(self.run_directory).exists():
            issues.append("Run directory does not exist")

        if self.training_progress < 0 or self.training_progress > 1:
            issues.append("Training progress must be between 0 and 1")

        return issues


class SessionStateManager:
    """Manager class for handling session state operations."""

    @staticmethod
    def initialize() -> None:
        """Initialize session state if not already present."""
        if "app_state" not in st.session_state:
            st.session_state.app_state = SessionState()

        # Ensure backward compatibility with existing code
        SessionStateManager._sync_legacy_state()

    @staticmethod
    def get() -> SessionState:
        """Get current session state."""
        if "app_state" not in st.session_state:
            SessionStateManager.initialize()
        return st.session_state.app_state

    @staticmethod
    def update(updates: dict[str, Any]) -> None:
        """Update session state with new values."""
        state = SessionStateManager.get()
        for key, value in updates.items():
            if hasattr(state, key):
                setattr(state, key, value)
        state.last_updated = datetime.now()

    @staticmethod
    def _sync_legacy_state() -> None:
        """Sync with legacy session state variables for compatibility."""
        state = SessionStateManager.get()

        # Sync with legacy variables
        if "config_path" in st.session_state:
            state.config_path = st.session_state.config_path
        if "run_directory" in st.session_state:
            state.run_directory = st.session_state.run_directory
        if "current_page" in st.session_state:
            state.current_page = st.session_state.current_page
        if "theme" in st.session_state:
            state.theme = st.session_state.theme

        # Update legacy variables to match state
        st.session_state.config_path = state.config_path
        st.session_state.run_directory = state.run_directory
        st.session_state.current_page = state.current_page
        st.session_state.theme = state.theme

    @staticmethod
    def save_to_file(filepath: Path) -> bool:
        """Save session state to file."""
        try:
            state = SessionStateManager.get()
            with open(filepath, "w") as f:
                json.dump(state.to_dict(), f, indent=2)
            return True
        except Exception as e:
            st.error(f"Failed to save session state: {e}")
            return False

    @staticmethod
    def load_from_file(filepath: Path) -> bool:
        """Load session state from file."""
        try:
            if not filepath.exists():
                return False

            with open(filepath) as f:
                data = json.load(f)

            state = SessionState.from_dict(data)
            st.session_state.app_state = state
            SessionStateManager._sync_legacy_state()
            return True
        except Exception as e:
            st.error(f"Failed to load session state: {e}")
            return False
