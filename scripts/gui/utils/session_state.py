"""
Session state management for the CrackSeg application.

This module provides structured session state management with
serialization, validation, and backward compatibility.
"""

import json
import threading
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

    # Process lifecycle state (NEW for subtask 5.6)
    process_pid: int | None = None
    process_state: str = (
        "idle"  # idle, starting, running, stopping, completed, failed, aborted
    )
    process_start_time: float | None = None
    process_command: list[str] = field(default_factory=list)
    process_working_dir: str | None = None
    process_return_code: int | None = None
    process_error_message: str | None = None
    process_memory_usage: dict[str, float] = field(default_factory=dict)

    # Log streaming state (NEW for subtask 5.6)
    log_streaming_active: bool = False
    log_buffer_size: int = 0
    log_last_update: datetime | None = None
    recent_logs: list[dict[str, Any]] = field(default_factory=list)
    hydra_run_dir: str | None = None

    # Training statistics extracted from logs (NEW for subtask 5.6)
    training_epoch: int | None = None
    training_loss: float | None = None
    training_learning_rate: float | None = None
    validation_metrics: dict[str, float] = field(default_factory=dict)

    # Model state
    model_loaded: bool = False
    model_architecture: str | None = None
    model_parameters: dict[str, Any] | None = None
    current_model: Any | None = (
        None  # Stores the actual PyTorch model instance
    )
    model_summary: dict[str, Any] = field(default_factory=dict)
    model_device: str | None = None
    architecture_diagram_path: str | None = (
        None  # Path to generated architecture diagram
    )

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

    # Thread safety for session state updates (NEW for subtask 5.6)
    _update_lock: threading.Lock = field(default_factory=threading.Lock)

    def to_dict(self) -> dict[str, Any]:
        """Convert session state to dictionary for serialization."""
        result = {}
        for key, value in self.__dict__.items():
            # Skip the lock object and other non-serializable items
            if key.startswith("_") or key in ["current_model"]:
                continue
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
        if "log_last_update" in data and isinstance(
            data["log_last_update"], str
        ):
            data["log_last_update"] = datetime.fromisoformat(
                data["log_last_update"]
            )

        return cls(
            **{k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        )

    def update_config(
        self, config_path: str, config_data: dict[str, Any] | None = None
    ) -> None:
        """Update configuration state."""
        with self._update_lock:
            self.config_path = config_path
            self.config_data = config_data
            self.config_loaded = config_data is not None
            self.last_updated = datetime.now()

    def update_training_progress(
        self, progress: float, metrics: dict[str, float] | None = None
    ) -> None:
        """Update training progress and metrics."""
        with self._update_lock:
            self.training_progress = max(0.0, min(1.0, progress))
            if metrics:
                self.training_metrics.update(metrics)
            self.last_updated = datetime.now()

    def set_training_active(self, active: bool) -> None:
        """Set training active state."""
        with self._update_lock:
            was_active = self.training_active
            self.training_active = active
            if not active:
                self.training_progress = 0.0
            self.last_updated = datetime.now()

            # Notify TensorBoard lifecycle manager of state change
            if was_active != active:
                self._notify_tensorboard_lifecycle_change()

    def _notify_tensorboard_lifecycle_change(self) -> None:
        """Notify TensorBoard lifecycle manager of training state changes."""
        try:
            # Import here to avoid circular imports
            from pathlib import Path

            from scripts.gui.utils.tb_manager import (
                get_global_lifecycle_manager,
            )

            lifecycle_manager = get_global_lifecycle_manager()

            # Determine training state for lifecycle manager
            if self.training_active:
                training_state = "running"
            elif self.process_state in ["completed", "failed", "aborted"]:
                training_state = self.process_state
            else:
                training_state = "idle"

            # Get log directory if available
            log_dir = None
            if self.run_directory:
                log_dir = Path(self.run_directory) / "logs" / "tensorboard"

            # Handle state change
            lifecycle_manager.handle_training_state_change(
                training_state=training_state,
                log_dir=log_dir,
            )

        except ImportError:
            # TensorBoard lifecycle not available, continue without
            # notification
            pass
        except Exception:
            # Log error but don't break training state updates
            pass

    # NEW methods for subtask 5.6: Process lifecycle management
    def update_process_state(
        self,
        state: str,
        pid: int | None = None,
        command: list[str] | None = None,
        start_time: float | None = None,
        working_dir: str | None = None,
        return_code: int | None = None,
        error_message: str | None = None,
        memory_usage: dict[str, float] | None = None,
    ) -> None:
        """Update process lifecycle state information.

        Args:
            state: Process state (idle, starting, running, stopping,
                completed, failed, aborted)
            pid: Process ID if available
            command: Command line arguments
            start_time: Process start timestamp
            working_dir: Working directory path
            return_code: Process exit code
            error_message: Error message if process failed
            memory_usage: Memory usage statistics
        """
        with self._update_lock:
            self.process_state = state
            if pid is not None:
                self.process_pid = pid
            if command is not None:
                self.process_command = command
            if start_time is not None:
                self.process_start_time = start_time
            if working_dir is not None:
                self.process_working_dir = working_dir
            if return_code is not None:
                self.process_return_code = return_code
            if error_message is not None:
                self.process_error_message = error_message
            if memory_usage is not None:
                self.process_memory_usage = memory_usage

            # Update training_active to match process state
            if state in ["running"]:
                self.training_active = True
            elif state in ["idle", "completed", "failed", "aborted"]:
                self.training_active = False

            self.last_updated = datetime.now()

            # Notify TensorBoard lifecycle manager of process state change
            self._notify_tensorboard_lifecycle_change()

    def update_log_streaming_state(
        self,
        active: bool,
        buffer_size: int | None = None,
        recent_logs: list[dict[str, Any]] | None = None,
        hydra_run_dir: str | None = None,
    ) -> None:
        """Update log streaming state information.

        Args:
            active: Whether log streaming is currently active
            buffer_size: Current size of log buffer
            recent_logs: List of recent log entries
            hydra_run_dir: Path to Hydra run directory
        """
        with self._update_lock:
            self.log_streaming_active = active
            if buffer_size is not None:
                self.log_buffer_size = buffer_size
            if recent_logs is not None:
                # Keep only last 50 log entries to prevent memory bloat
                self.recent_logs = recent_logs[-50:]
            if hydra_run_dir is not None:
                self.hydra_run_dir = hydra_run_dir

            self.log_last_update = datetime.now()
            self.last_updated = datetime.now()

    def update_training_stats_from_logs(
        self,
        epoch: int | None = None,
        loss: float | None = None,
        learning_rate: float | None = None,
        validation_metrics: dict[str, float] | None = None,
    ) -> None:
        """Update training statistics extracted from log parsing.

        Args:
            epoch: Current training epoch
            loss: Current training loss
            learning_rate: Current learning rate
            validation_metrics: Validation metrics dictionary
        """
        with self._update_lock:
            if epoch is not None:
                self.training_epoch = epoch
            if loss is not None:
                self.training_loss = loss
            if learning_rate is not None:
                self.training_learning_rate = learning_rate
            if validation_metrics is not None:
                self.validation_metrics.update(validation_metrics)

            self.last_updated = datetime.now()

    def reset_process_state(self) -> None:
        """Reset all process-related state to initial values."""
        with self._update_lock:
            self.process_pid = None
            self.process_state = "idle"
            self.process_start_time = None
            self.process_command = []
            self.process_working_dir = None
            self.process_return_code = None
            self.process_error_message = None
            self.process_memory_usage = {}

            self.log_streaming_active = False
            self.log_buffer_size = 0
            self.recent_logs = []
            self.hydra_run_dir = None

            self.training_active = False
            self.training_progress = 0.0
            self.training_epoch = None
            self.training_loss = None
            self.training_learning_rate = None
            self.validation_metrics = {}

            self.last_updated = datetime.now()

    def add_notification(self, message: str) -> None:
        """Add a notification message."""
        with self._update_lock:
            self.notifications.append(
                f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
            )
            # Keep only last 10 notifications
            self.notifications = self.notifications[-10:]
            self.last_updated = datetime.now()

    def clear_notifications(self) -> None:
        """Clear all notifications."""
        with self._update_lock:
            self.notifications.clear()
            self.last_updated = datetime.now()

    def is_ready_for_training(self) -> bool:
        """Check if application is ready for training."""
        return self.config_loaded and self.run_directory is not None

    def is_ready_for_results(self) -> bool:
        """Check if results can be displayed."""
        return self.results_available or self.run_directory is not None

    def is_process_running(self) -> bool:
        """Check if a training process is currently running."""
        return self.process_state in ["starting", "running"]

    def get_process_status_summary(self) -> dict[str, Any]:
        """Get a summary of current process status for UI display."""
        return {
            "state": self.process_state,
            "pid": self.process_pid,
            "active": self.is_process_running(),
            "start_time": self.process_start_time,
            "memory_usage": self.process_memory_usage,
            "error": self.process_error_message,
            "log_streaming": self.log_streaming_active,
            "hydra_dir": self.hydra_run_dir,
        }

    def set_theme(self, theme: str) -> None:
        """Update theme setting."""
        with self._update_lock:
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

        # Validate process state consistency
        if self.process_state == "running" and not self.training_active:
            issues.append("Process state and training_active are inconsistent")

        if self.training_active and self.process_state not in [
            "running",
            "starting",
        ]:
            issues.append("Training active but process state is not running")

        return issues


class SessionStateManager:
    """Facade for interacting with Streamlit's session state in a structured
    way."""

    _STATE_KEY = "_crackseg_state"

    @staticmethod
    def _get_state_container(
        session_state_proxy: dict[str, Any] | None = None,
    ) -> dict[str, Any] | Any:
        """Get the container for the app state, allowing for DI."""
        if session_state_proxy is not None:
            return session_state_proxy
        # In the real application, this will be st.session_state
        return st.session_state

    @staticmethod
    def initialize(
        session_state_proxy: dict[str, Any] | Any | None = None,
    ) -> None:
        """Initialize session state if not already present."""
        state_container = SessionStateManager._get_state_container(
            session_state_proxy
        )
        if SessionStateManager._STATE_KEY not in state_container:
            state_container[SessionStateManager._STATE_KEY] = SessionState()

    @staticmethod
    def get(
        session_state_proxy: dict[str, Any] | Any | None = None,
    ) -> SessionState:
        """Get the current session state."""
        state_container = SessionStateManager._get_state_container(
            session_state_proxy
        )
        if SessionStateManager._STATE_KEY not in state_container:
            SessionStateManager.initialize(state_container)
        return state_container[SessionStateManager._STATE_KEY]

    @staticmethod
    def update(
        updates: dict[str, Any],
        session_state_proxy: dict[str, Any] | Any | None = None,
    ) -> None:
        """Update session state with a dictionary of values."""
        state = SessionStateManager.get(session_state_proxy)
        for key, value in updates.items():
            if hasattr(state, key):
                setattr(state, key, value)

    @staticmethod
    def notify_change(change_type: str) -> None:
        """Notify of a state change for debugging/logging."""
        state = SessionStateManager.get()
        state.add_notification(f"State change: {change_type}")
        state.last_updated = datetime.now()

    # NEW methods for subtask 5.6: Process lifecycle integration
    @staticmethod
    def update_from_process_info(process_info: Any) -> None:
        """Update session state from ProcessInfo object.

        Args:
            process_info: ProcessInfo object from ProcessManager
        """
        state = SessionStateManager.get()

        # Map ProcessState enum to string if needed
        process_state = getattr(process_info, "state", None)
        if process_state is not None and hasattr(process_state, "value"):
            state_str = process_state.value
        elif process_state is not None:
            state_str = str(process_state)
        else:
            state_str = "unknown"

        state.update_process_state(
            state=state_str,
            pid=getattr(process_info, "pid", None),
            command=getattr(process_info, "command", []),
            start_time=getattr(process_info, "start_time", None),
            working_dir=(
                str(getattr(process_info, "working_directory", ""))
                if getattr(process_info, "working_directory", None)
                else None
            ),
            return_code=getattr(process_info, "return_code", None),
            error_message=getattr(process_info, "error_message", None),
        )

    @staticmethod
    def update_from_log_stream_info(
        active: bool,
        buffer_size: int | None = None,
        recent_logs: list[dict[str, Any]] | None = None,
        hydra_run_dir: str | None = None,
    ) -> None:
        """Update session state from log streaming information.

        Args:
            active: Whether log streaming is active
            buffer_size: Current buffer size
            recent_logs: Recent log entries
            hydra_run_dir: Hydra run directory path
        """
        state = SessionStateManager.get()
        state.update_log_streaming_state(
            active=active,
            buffer_size=buffer_size,
            recent_logs=recent_logs,
            hydra_run_dir=hydra_run_dir,
        )

    @staticmethod
    def extract_training_stats_from_logs(logs: list[dict[str, Any]]) -> None:
        """Extract and update training statistics from log entries.

        Args:
            logs: List of log entries to parse for training statistics
        """
        state = SessionStateManager.get()

        # Simple patterns for extracting training metrics from logs
        # This can be enhanced with more sophisticated parsing
        latest_stats = {}

        for log_entry in logs[-10:]:  # Check last 10 logs
            message = log_entry.get("message", "").lower()

            # Extract epoch information
            if "epoch" in message:
                try:
                    # Look for patterns like "epoch: 5" or "epoch 5/100"
                    import re

                    epoch_match = re.search(r"epoch[:\s]+(\d+)", message)
                    if epoch_match:
                        latest_stats["epoch"] = int(epoch_match.group(1))
                except (ValueError, AttributeError):
                    pass

            # Extract loss information
            if "loss" in message:
                try:
                    import re

                    loss_match = re.search(
                        r"loss[:\s]+([0-9]+\.?[0-9]*)", message
                    )
                    if loss_match:
                        latest_stats["loss"] = float(loss_match.group(1))
                except (ValueError, AttributeError):
                    pass

            # Extract learning rate
            if "lr" in message or "learning_rate" in message:
                try:
                    import re

                    lr_match = re.search(
                        r"(?:lr|learning_rate)[:\s]+([0-9]+\.?[0-9]*(?:e-?\d+)?)",
                        message,
                    )
                    if lr_match:
                        latest_stats["learning_rate"] = float(
                            lr_match.group(1)
                        )
                except (ValueError, AttributeError):
                    pass

        # Update session state with extracted statistics
        if latest_stats:
            state.update_training_stats_from_logs(
                epoch=latest_stats.get("epoch"),
                loss=latest_stats.get("loss"),
                learning_rate=latest_stats.get("learning_rate"),
            )

    @staticmethod
    def reset_training_session() -> None:
        """Reset all training and process related state."""
        state = SessionStateManager.get()
        state.reset_process_state()
        SessionStateManager.notify_change("training_session_reset")

    @staticmethod
    def _sync_legacy_state() -> None:
        """Sync with legacy session state variables for compatibility."""
        if "app_state" not in st.session_state:
            return

        state = st.session_state.app_state

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
    def save_to_file(
        filepath: Path, session_state_proxy: dict[str, Any] | Any | None = None
    ) -> bool:
        """Save current session state to a JSON file."""
        state = SessionStateManager.get(session_state_proxy)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(state.to_dict(), f, indent=4)
            return True
        except (OSError, TypeError) as e:
            st.error(f"Error saving session state: {e}")
            return False

    @staticmethod
    def load_from_file(
        filepath: Path, session_state_proxy: dict[str, Any] | Any | None = None
    ) -> bool:
        """Load session state from a JSON file."""
        state_container = SessionStateManager._get_state_container(
            session_state_proxy
        )
        if not filepath.exists():
            return False
        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
            state_container[SessionStateManager._STATE_KEY] = (
                SessionState.from_dict(data)
            )
            return True
        except (OSError, json.JSONDecodeError, TypeError) as e:
            st.error(f"Error loading session state: {e}")
            return False
