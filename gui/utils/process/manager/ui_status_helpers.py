"""
UI helper functions for status updates in Streamlit components. This
module provides convenient functions for Streamlit components to
display status updates, manage callbacks, and format status
information in a user-friendly way.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import streamlit as st

from .status_integration import (
    add_status_update_callback,
    get_comprehensive_status,
    initialize_status_integration,
    remove_status_update_callback,
)
from .status_updates import StatusUpdate, StatusUpdateType


def initialize_ui_status_system() -> bool:
    """
    Initialize the status update system for UI components. Should be
    called once when the Streamlit app starts. Returns: True if
    initialization successful, False otherwise
    """
    if "status_system_initialized" not in st.session_state:
        success = initialize_status_integration()
        st.session_state.status_system_initialized = success
        return success

    return st.session_state.status_system_initialized


def display_training_status() -> None:
    """
    Display comprehensive training status in Streamlit UI. Creates a
    formatted status display with process state, streaming info, and
    recent updates.
    """
    try:
        status = get_comprehensive_status()

        # Main status indicators
        col1, col2, col3 = st.columns(3)

        with col1:
            process_status = status.get("process", {})
            is_running = process_status.get("is_running", False)

            if is_running:
                st.success("🟢 Training Running")
                if process_status.get("pid"):
                    st.caption(f"PID: {process_status['pid']}")
            else:
                st.info("⚪ Training Stopped")

        with col2:
            streaming_status = status.get("streaming", {})
            is_streaming = streaming_status.get("is_streaming", False)

            if is_streaming:
                st.success("📡 Logs Streaming")
                logs_count = streaming_status.get("total_logs_processed", 0)
                st.caption(f"Logs: {logs_count}")
            else:
                st.info("📡 No Streaming")

        with col3:
            manager_active = status.get("manager_active", False)

            if manager_active:
                st.success("⚙️ Status Manager Active")
                callbacks = status.get("active_callbacks", 0)
                st.caption(f"Callbacks: {callbacks}")
            else:
                st.warning("⚙️ Status Manager Inactive")

        # Detailed status in expander
        with st.expander("📊 Detailed Status", expanded=False):
            st.json(status)

        # Recent errors if any
        recent_errors = status.get("recent_errors", [])
        if recent_errors:
            st.error("⚠️ Recent Errors")
            for error in recent_errors[-3:]:  # Show last 3 errors
                with st.expander(
                    f"Error: {error.get('error_type', 'Unknown')}",
                    expanded=False,
                ):
                    st.code(error.get("message", "No message"))
                    st.caption(f"Time: {error.get('timestamp', 'Unknown')}")

    except Exception as e:
        st.error(f"Failed to display status: {e}")


def display_status_updates_feed(max_updates: int = 20) -> None:
    """
    Display a live feed of status updates. Args: max_updates: Maximum
    number of updates to display
    """
    from .status_updates import get_status_update_manager

    try:
        status_manager = get_status_update_manager()
        recent_updates = status_manager.get_recent_updates(count=max_updates)

        if not recent_updates:
            st.info("No status updates yet")
            return

        st.subheader("📢 Status Updates Feed")

        for update in reversed(recent_updates):  # Show newest first
            priority_icons = {1: "ℹ️", 2: "📝", 3: "⚠️", 4: "🚨"}
            icon = priority_icons.get(update.priority, "📝")

            # Format timestamp
            time_str = update.timestamp.strftime("%H:%M:%S")

            # Create update display
            with st.container():
                col1, col2 = st.columns([1, 4])

                with col1:
                    st.caption(f"{icon} {time_str}")
                    st.caption(f"**{update.source}**")

                with col2:
                    update_title = _format_update_title(update)
                    st.write(f"**{update_title}**")

                    if update.data:
                        with st.expander("Details", expanded=False):
                            st.json(update.data)

                st.divider()

    except Exception as e:
        st.error(f"Failed to display updates feed: {e}")


def create_status_callback_for_ui() -> None:
    """
    Create and register a status callback that updates Streamlit state.
    This allows UI components to react to status changes by checking
    session state variables.
    """

    def ui_status_callback(update: StatusUpdate) -> None:
        """Handle status updates for UI components."""
        try:
            # Update session state with latest status info
            if "latest_status_update" not in st.session_state:
                st.session_state.latest_status_update = {}

            st.session_state.latest_status_update = {
                "type": update.update_type.value,
                "timestamp": update.timestamp.isoformat(),
                "data": update.data,
                "source": update.source,
                "priority": update.priority,
            }

            # Update specific status flags for easy checking
            if update.update_type == StatusUpdateType.PROCESS_STARTED:
                st.session_state.training_running = True
                st.session_state.last_process_start = update.timestamp

            elif update.update_type == StatusUpdateType.PROCESS_STOPPED:
                st.session_state.training_running = False
                st.session_state.last_process_stop = update.timestamp

            elif update.update_type == StatusUpdateType.PROCESS_ERROR:
                st.session_state.training_error = True
                st.session_state.last_error = update.data

            elif update.update_type == StatusUpdateType.LOG_RECEIVED:
                if "log_count" not in st.session_state:
                    st.session_state.log_count = 0
                st.session_state.log_count += 1

            # Force UI rerun if needed (careful with this)
            # st.rerun()  # Uncomment if immediate UI updates are needed

        except Exception as e:
            print(f"Error in UI status callback: {e}")

    # Register the callback
    add_status_update_callback(ui_status_callback)

    # Store callback reference for cleanup
    if "ui_status_callback" not in st.session_state:
        st.session_state.ui_status_callback = ui_status_callback


def cleanup_ui_status_callback() -> None:
    """Clean up the UI status callback."""
    if "ui_status_callback" in st.session_state:
        callback = st.session_state.ui_status_callback
        remove_status_update_callback(callback)
        del st.session_state.ui_status_callback


def get_training_status_indicator() -> tuple[str, str]:
    """
    Get a simple training status indicator for UI display. Returns: Tuple
    of (status_text, status_color)
    """
    try:
        status = get_comprehensive_status()
        process_status = status.get("process", {})

        is_running = process_status.get("is_running", False)
        has_errors = len(process_status.get("recent_errors", [])) > 0

        if is_running:
            return ("🟢 Training Active", "success")
        elif has_errors:
            return ("🔴 Training Error", "error")
        else:
            return ("⚪ Training Idle", "info")

    except Exception:
        return ("❓ Status Unknown", "warning")


def display_metrics_summary() -> None:
    """Display training metrics summary if available."""
    try:
        from .status_updates import get_status_update_manager

        status_manager = get_status_update_manager()
        recent_updates = status_manager.get_recent_updates(count=50)

        # Find latest metrics update
        metrics_update = None
        for update in reversed(recent_updates):
            if update.update_type == StatusUpdateType.METRICS_UPDATED:
                metrics_update = update
                break

        if metrics_update and metrics_update.data.get("metrics"):
            st.subheader("📈 Latest Metrics")
            metrics = metrics_update.data["metrics"]

            # Display metrics in columns
            metric_cols = st.columns(len(metrics))
            for i, (name, value) in enumerate(metrics.items()):
                with metric_cols[i]:
                    st.metric(
                        label=name.replace("_", " ").title(),
                        value=(
                            f"{value:.4f}"
                            if isinstance(value, float)
                            else str(value)
                        ),
                    )

            # Show timestamp
            st.caption(
                f"Updated: {metrics_update.timestamp.strftime('%H:%M:%S')}"
            )
        else:
            st.info("No metrics available yet")

    except Exception as e:
        st.error(f"Failed to display metrics: {e}")


def _format_update_title(update: StatusUpdate) -> str:
    """
    Format a status update title for display. Args: update: Status update
    to format Returns: Formatted title string
    """
    type_titles = {
        StatusUpdateType.PROCESS_STARTED: "Training Started",
        StatusUpdateType.PROCESS_STOPPED: "Training Stopped",
        StatusUpdateType.PROCESS_ERROR: "Training Error",
        StatusUpdateType.LOG_RECEIVED: "Log Message",
        StatusUpdateType.METRICS_UPDATED: "Metrics Updated",
        StatusUpdateType.PROGRESS_UPDATED: "Progress Update",
        StatusUpdateType.SESSION_STATE_CHANGED: "Session State Changed",
        StatusUpdateType.ABORT_INITIATED: "Abort Initiated",
        StatusUpdateType.ABORT_COMPLETED: "Abort Completed",
    }

    base_title = type_titles.get(update.update_type, update.update_type.value)

    # Add specific details based on update type
    if update.update_type == StatusUpdateType.LOG_RECEIVED:
        level = update.data.get("level", "").upper()
        if level:
            base_title = f"{base_title} ({level})"

    elif update.update_type == StatusUpdateType.METRICS_UPDATED:
        metrics = update.data.get("metrics", {})
        if metrics:
            metric_names = list(metrics.keys())[
                :2
            ]  # Show first 2 metric names
            if metric_names:
                base_title = f"{base_title}: {', '.join(metric_names)}"

    return base_title


def check_status_changes_since(last_check: datetime) -> dict[str, Any]:
    """
    Check for status changes since a given timestamp. Args: last_check:
    Timestamp to check changes since Returns: Dictionary with change
    information
    """
    try:
        from .status_updates import get_status_update_manager

        status_manager = get_status_update_manager()
        recent_updates = status_manager.get_recent_updates(since=last_check)

        changes = {
            "has_changes": len(recent_updates) > 0,
            "update_count": len(recent_updates),
            "latest_update": (
                recent_updates[-1].to_dict() if recent_updates else None
            ),
            "process_changes": any(
                "process" in u.update_type.value for u in recent_updates
            ),
            "error_updates": [
                u
                for u in recent_updates
                if u.priority >= 3 and "error" in u.update_type.value
            ],
        }

        return changes

    except Exception as e:
        return {
            "has_changes": False,
            "error": str(e),
            "update_count": 0,
        }
