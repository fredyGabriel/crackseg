r"""
Confirmation dialog renderer for the CrackSeg application. This module
handles the rendering, styling, and HTML generation for\ confirmation
dialogs. Separated from core logic to maintain file size limits and
single responsibility.
"""

import streamlit as st

from crackseg.utils.logging import get_logger
from gui.components.confirmation_dialog import (
    ConfirmationDialog,
    track_performance_decorator,
)
from gui.utils.performance_optimizer import inject_css_once

logger = get_logger(__name__)


class OptimizedConfirmationDialog:
    """High-performance confirmation dialog with caching and optimization."""

    _CSS_CONTENT = """
<style> .crackseg-confirmation-dialog { position: fixed; top: 0; left:
0; width: 100%; height: 100%; background: rgba(0, 0, 0, 0.6); display:
flex; justify-content: center; align-items: center; z-index: 9999;
font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
.crackseg-confirmation-modal { background: white; border-radius: 12px;
padding: 2rem; max-width: 500px; width: 90%; max-height: 80vh;
overflow-y: auto; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
animation: dialogSlideIn 0.3s ease-out; } @keyframes dialogSlideIn {
from { opacity: 0; transform: translateY(-20px) scale(0.95); } to {
opacity: 1; transform: translateY(0) scale(1); } }
.crackseg-confirmation-header { display: flex; align-items: center;
margin-bottom: 1.5rem; } .crackseg-confirmation-icon { width: 48px;
height: 48px; border-radius: 50%; display: flex; align-items: center;
justify-content: center; margin-right: 1rem; font-size: 1.5rem;
font-weight: bold; } .crackseg-confirmation-icon.low { background:
#e3f2fd; color: #1976d2; } .crackseg-confirmation-icon.medium {
background: #fff3e0; color: #f57c00; }
.crackseg-confirmation-icon.high { background: #ffebee; color:
#d32f2f; } .crackseg-confirmation-title { font-size: 1.25rem;
font-weight: 600; color: #2c3e50; margin: 0; }
.crackseg-confirmation-message { font-size: 1rem; color: #34495e;
line-height: 1.5; margin-bottom: 1rem; }
.crackseg-confirmation-warning { background: #fff3cd; border: 1px
solid #ffeaa7; border-radius: 6px; padding: 0.75rem; margin-bottom:
1rem; font-size: 0.9rem; color: #856404; }
.crackseg-confirmation-input { margin-bottom: 1rem; }
.crackseg-confirmation-input label { display: block; font-size:
0.9rem; color: #2c3e50; margin-bottom: 0.5rem; font-weight: 500; }
.crackseg-confirmation-input input { width: 100%; padding: 0.75rem;
border: 2px solid #e0e0e0; border-radius: 6px; font-size: 1rem;
box-sizing: border-box; transition: border-color 0.3s ease; }
.crackseg-confirmation-input input:focus { outline: none;
border-color: #3498db; } .crackseg-confirmation-input input.error {
border-color: #e74c3c; } .crackseg-confirmation-buttons { display:
flex; gap: 1rem; justify-content: flex-end; margin-top: 1.5rem; }
.crackseg-confirmation-button { padding: 0.75rem 1.5rem; border: none;
border-radius: 6px; font-size: 1rem; font-weight: 500; cursor:
pointer; transition: all 0.3s ease; min-width: 100px; }
.crackseg-confirmation-button.cancel { background: #f8f9fa; color:
#6c757d; border: 2px solid #e9ecef; }
.crackseg-confirmation-button.cancel:hover { background: #e9ecef;
color: #495057; } .crackseg-confirmation-button.confirm { background:
#3498db; color: white; } .crackseg-confirmation-button.confirm:hover {
background: #2980b9; } .crackseg-confirmation-button.confirm.medium {
background: #f39c12; }
.crackseg-confirmation-button.confirm.medium:hover { background:
#e67e22; } .crackseg-confirmation-button.confirm.high { background:
#e74c3c; } .crackseg-confirmation-button.confirm.high:hover {
background: #c0392b; } .crackseg-confirmation-button:disabled {
opacity: 0.6; cursor: not-allowed; } @media (max-width: 768px) {
.crackseg-confirmation-modal { width: 95%; padding: 1.5rem; }
.crackseg-confirmation-buttons { flex-direction: column; }
.crackseg-confirmation-button { width: 100%; } } </style>
"""

    @staticmethod
    def _ensure_css_injected() -> None:
        """Ensure CSS is injected only once."""
        inject_css_once(
            "crackseg_confirmation_dialog",
            OptimizedConfirmationDialog._CSS_CONTENT,
        )

    @staticmethod
    def _get_icon_for_danger_level(danger_level: str) -> str:
        """Get icon based on danger level."""
        icons = {"low": "ℹ️", "medium": "⚠️", "high": "⚠️"}
        return icons.get(danger_level, "❓")

    @staticmethod
    @track_performance_decorator("confirmation_dialog_render")
    def show_confirmation_dialog(
        dialog: ConfirmationDialog,
        component_id: str = "confirmation_dialog",
        session_key: str = "confirmation_dialog_state",
    ) -> str | None:
        """
        Display confirmation dialog and return user action.

        Args:
            dialog: ConfirmationDialog instance
            component_id: Unique component identifier
            session_key: Session state key

        Returns:
            "confirmed" if user confirmed, "cancelled" if cancelled,
            None if pending
        """
        # Ensure CSS is injected
        OptimizedConfirmationDialog._ensure_css_injected()

        # Initialize session state
        if session_key not in st.session_state:
            st.session_state[session_key] = {
                "active": False,
                "dialog": None,
                "user_input": "",
                "result": None,
            }

        dialog_state = st.session_state[session_key]

        # If dialog is not active, return None
        if not dialog_state["active"]:
            return None

        # Set current dialog
        dialog_state["dialog"] = dialog

        # Create dialog HTML
        icon = OptimizedConfirmationDialog._get_icon_for_danger_level(
            dialog.danger_level
        )

        modal_html = f"""
        <div class="crackseg-confirmation-dialog" id="{component_id}">
            <div class="crackseg-confirmation-modal">
                <div class="crackseg-confirmation-header">
                    <div class="crackseg-confirmation-icon {
            dialog.danger_level
        }">
                        {icon}
                    </div>
                    <h3 class="crackseg-confirmation-title">{dialog.title}</h3>
                </div>
                <div class="crackseg-confirmation-message">
                    {dialog.message}
                </div>
                """

        if dialog.warning_text:
            modal_html += f"""
                <div class="crackseg-confirmation-warning">
                    <strong>Warning:</strong> {dialog.warning_text}
                </div>
                """

        modal_html += "</div></div>"

        # Display the modal
        st.markdown(modal_html, unsafe_allow_html=True)

        # Handle typing confirmation if required
        if dialog.requires_typing:
            st.markdown("**Please type the confirmation phrase to proceed:**")
            user_input = st.text_input(
                f"Type '{dialog.confirmation_phrase}' to confirm:",
                key=f"{session_key}_input",
                placeholder=dialog.confirmation_phrase,
            )
            dialog_state["user_input"] = user_input

            # Check if input matches
            input_valid = (
                dialog.confirmation_phrase is not None
                and user_input.strip().upper()
                == dialog.confirmation_phrase.upper()
            )
        else:
            input_valid = True

        # Action buttons
        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                dialog.cancel_text,
                key=f"{session_key}_cancel",
                use_container_width=True,
            ):
                dialog_state["active"] = False
                dialog_state["result"] = "cancelled"
                st.rerun()

        with col2:
            if st.button(
                dialog.confirm_text,
                key=f"{session_key}_confirm",
                disabled=not input_valid,
                use_container_width=True,
                type="primary",
            ):
                dialog_state["active"] = False
                dialog_state["result"] = "confirmed"
                st.rerun()

        return dialog_state.get("result")

    @staticmethod
    def activate_dialog(
        dialog: ConfirmationDialog,
        session_key: str = "confirmation_dialog_state",
    ) -> None:
        """Activate a confirmation dialog."""
        if session_key not in st.session_state:
            st.session_state[session_key] = {}

        st.session_state[session_key]["active"] = True
        st.session_state[session_key]["dialog"] = dialog
        st.session_state[session_key]["result"] = None
        st.session_state[session_key]["user_input"] = ""

    @staticmethod
    def is_dialog_active(
        session_key: str = "confirmation_dialog_state",
    ) -> bool:
        """Check if a dialog is currently active."""
        return session_key in st.session_state and st.session_state[
            session_key
        ].get("active", False)
