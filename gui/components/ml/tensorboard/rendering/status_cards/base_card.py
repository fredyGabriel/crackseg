"""Base status card component with common functionality.

Provides the foundation for all TensorBoard status cards with shared
rendering utilities and data structures.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import streamlit as st


@dataclass
class StatusInfo:
    """Status information container for cards."""

    status: str
    details: list[str]
    color: str


class BaseStatusCard(ABC):
    """Base class for TensorBoard status cards."""

    # Color mapping for status indicators
    COLOR_STYLES = {
        "success": "background-color: rgba(0, 255, 0, 0.1); "
        "border-left: 4px solid #00ff00;",
        "info": "background-color: rgba(0, 123, 255, 0.1); "
        "border-left: 4px solid #007bff;",
        "warning": "background-color: rgba(255, 193, 7, 0.1); "
        "border-left: 4px solid #ffc107;",
        "error": "background-color: rgba(220, 53, 69, 0.1); "
        "border-left: 4px solid #dc3545;",
    }

    def __init__(self, title: str) -> None:
        """Initialize base status card.

        Args:
            title: Display title for the card.
        """
        self.title = title

    @abstractmethod
    def collect_status_info(
        self, manager: Any, session_manager: Any
    ) -> StatusInfo:
        """Collect status information for this card.

        Args:
            manager: TensorBoard manager instance.
            session_manager: Session state manager.

        Returns:
            StatusInfo containing status, details, and color.
        """
        pass

    def render(self, manager: Any, session_manager: Any) -> None:
        """Render the status card.

        Args:
            manager: TensorBoard manager instance.
            session_manager: Session state manager.
        """
        status_info = self.collect_status_info(manager, session_manager)
        self._render_card(status_info)

    def _render_card(self, status_info: StatusInfo) -> None:
        """Render the status card HTML.

        Args:
            status_info: Status information to display.
        """
        style = self.COLOR_STYLES.get(
            status_info.color, self.COLOR_STYLES["info"]
        )

        # Build detail HTML with proper line length
        detail_html = "".join(
            f'<div style="font-size: 11px; color: #666; margin: 2px 0;">'
            f"{detail}</div>"
            for detail in status_info.details
        )

        card_html = f"""
        <div style="padding: 12px; border-radius: 6px; margin: 4px 0; {style}">
            <div style="font-weight: bold; font-size: 14px;
                       margin-bottom: 6px;">
                {self.title}
            </div>
            <div style="font-size: 13px; margin-bottom: 4px;">
                {status_info.status}
            </div>
            {detail_html}
        </div>
        """

        st.markdown(card_html, unsafe_allow_html=True)
