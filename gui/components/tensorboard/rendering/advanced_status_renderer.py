"""Advanced status rendering for TensorBoard component.

This module provides comprehensive, real-time status indicators that display
process state, network connectivity, health monitoring, and resource usage
for TensorBoard instances using modular card components.

Features:
- Real-time process state indicators with visual feedback
- Network connectivity monitoring with latency measurements
- Health monitoring with diagnostic information
- Resource usage tracking (memory, CPU when available)
- Progressive status updates with detailed breakdown
- Error categorization and recovery suggestions
"""

import streamlit as st

from gui.utils.tb_manager import TensorBoardManager

from ..state.session_manager import SessionStateManager
from .diagnostics import ActionControls, DiagnosticPanel
from .status_cards import (
    HealthStatusCard,
    NetworkStatusCard,
    ProcessStatusCard,
    ResourceStatusCard,
)


def render_advanced_status_section(
    manager: TensorBoardManager,
    session_manager: SessionStateManager,
    show_refresh: bool = True,
    show_diagnostics: bool = True,
    compact_mode: bool = False,
) -> None:
    """Render advanced TensorBoard status section with comprehensive indicators

    Args:
        manager: TensorBoard manager instance.
        session_manager: Session state manager.
        show_refresh: Whether to show refresh button.
        show_diagnostics: Show detailed diagnostic information.
        compact_mode: Use compact layout for limited space.
    """
    if compact_mode:
        _render_compact_status(manager, session_manager, show_refresh)
    else:
        _render_full_status(
            manager, session_manager, show_refresh, show_diagnostics
        )


def _render_full_status(
    manager: TensorBoardManager,
    session_manager: SessionStateManager,
    show_refresh: bool,
    show_diagnostics: bool,
) -> None:
    """Render full advanced status display."""
    # Primary status indicators
    _render_primary_indicators(manager, session_manager)

    # Detailed status breakdown
    if show_diagnostics:
        diagnostic_panel = DiagnosticPanel()
        diagnostic_panel.render(manager, session_manager)

    # Quick actions
    if show_refresh:
        action_controls = ActionControls()
        action_controls.render(manager, session_manager)


def _render_compact_status(
    manager: TensorBoardManager,
    session_manager: SessionStateManager,
    show_refresh: bool,
) -> None:
    """Render compact status display for limited space."""
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        status_indicator = _get_simple_status_indicator(manager)
        st.markdown(status_indicator["text"])

    with col2:
        if show_refresh:
            action_controls = ActionControls()
            action_controls.render_compact(manager, session_manager)


def _render_primary_indicators(
    manager: TensorBoardManager, session_manager: SessionStateManager
) -> None:
    """Render primary status indicators with visual feedback."""
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    # Initialize status cards
    process_card = ProcessStatusCard()
    network_card = NetworkStatusCard()
    health_card = HealthStatusCard()
    resource_card = ResourceStatusCard()

    # Render cards in columns
    with col1:
        process_card.render(manager, session_manager)

    with col2:
        network_card.render(manager, session_manager)

    with col3:
        health_card.render(manager, session_manager)

    with col4:
        resource_card.render(manager, session_manager)


def _get_simple_status_indicator(
    manager: TensorBoardManager,
) -> dict[str, str]:
    """Get simple status indicator for compact display."""
    from gui.utils.tb_manager import TensorBoardState

    state = manager.info.state

    if state == TensorBoardState.RUNNING:
        return {"text": "🟢 **TensorBoard Running**", "color": "success"}
    elif state == TensorBoardState.STARTING:
        return {"text": "🟡 **Starting TensorBoard**", "color": "warning"}
    elif state == TensorBoardState.STOPPING:
        return {"text": "🟠 **Stopping TensorBoard**", "color": "warning"}
    elif state == TensorBoardState.FAILED:
        return {"text": "🔴 **TensorBoard Failed**", "color": "error"}
    else:
        return {"text": "⚪ **TensorBoard Stopped**", "color": "info"}


# Legacy compatibility functions
def render_status_cards(
    manager: TensorBoardManager, session_manager: SessionStateManager
) -> None:
    """Legacy compatibility function for status cards rendering.

    Args:
        manager: TensorBoard manager instance.
        session_manager: Session state manager.
    """
    _render_primary_indicators(manager, session_manager)


def render_diagnostic_panel(
    manager: TensorBoardManager, session_manager: SessionStateManager
) -> None:
    """Legacy compatibility function for diagnostic panel rendering.

    Args:
        manager: TensorBoard manager instance.
        session_manager: Session state manager.
    """
    diagnostic_panel = DiagnosticPanel()
    diagnostic_panel.render(manager, session_manager)


def render_action_controls(
    manager: TensorBoardManager, session_manager: SessionStateManager
) -> None:
    """Legacy compatibility function for action controls rendering.

    Args:
        manager: TensorBoard manager instance.
        session_manager: Session state manager.
    """
    action_controls = ActionControls()
    action_controls.render(manager, session_manager)
