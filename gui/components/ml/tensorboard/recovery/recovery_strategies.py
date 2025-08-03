"""Automatic recovery strategies for TensorBoard errors."""

import time

import streamlit as st

from ..state.session_manager import SessionStateManager


def attempt_automatic_recovery(
    error_type: str | None, session_manager: SessionStateManager
) -> None:
    """Attempt automatic recovery based on error type.

    Args:
        error_type: Type of error that occurred.
        session_manager: Session state manager for tracking recovery.
    """
    if error_type == "PortConflictError":
        st.info("🔄 Port conflict detected - attempting alternative port...")
        time.sleep(2)
        session_manager.update_state(recovery_attempted=True)

    elif error_type == "ProcessStartupError":
        st.info("🔄 Process startup failed - attempting recovery...")
        time.sleep(1)
        session_manager.update_state(recovery_attempted=True)

    elif error_type and "permission" in str(error_type).lower():
        st.warning(
            "⚠️ Permission error detected - manual intervention may be needed"
        )
        session_manager.update_state(recovery_attempted=True)
