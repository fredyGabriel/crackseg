"""Process status card for TensorBoard component."""

from typing import Any

from scripts.gui.components.tensorboard.utils.formatters import format_uptime
from scripts.gui.utils.tb_manager import TensorBoardState

from .base_card import BaseStatusCard, StatusInfo


class ProcessStatusCard(BaseStatusCard):
    """Status card for TensorBoard process information."""

    def __init__(self) -> None:
        """Initialize process status card."""
        super().__init__("🔧 Process")

    def collect_status_info(
        self, manager: Any, session_manager: Any
    ) -> StatusInfo:
        """Collect process status information.

        Args:
            manager: TensorBoard manager instance.
            session_manager: Session state manager.

        Returns:
            StatusInfo with process details.
        """
        info = manager.info
        state = info.state

        if state == TensorBoardState.RUNNING:
            uptime = info.get_uptime()
            uptime_str = format_uptime(uptime) if uptime else "Unknown"
            return StatusInfo(
                status="🟢 Running",
                details=[
                    f"PID: {info.pid}",
                    f"Port: {info.port}",
                    f"Uptime: {uptime_str}",
                ],
                color="success",
            )
        elif state == TensorBoardState.STARTING:
            attempts = info.startup_attempts
            return StatusInfo(
                status="🟡 Starting",
                details=[
                    f"Attempt: {attempts}",
                    "Initializing process...",
                ],
                color="warning",
            )
        elif state == TensorBoardState.STOPPING:
            return StatusInfo(
                status="🟠 Stopping",
                details=["Shutting down gracefully..."],
                color="warning",
            )
        elif state == TensorBoardState.FAILED:
            return StatusInfo(
                status="🔴 Failed",
                details=[
                    f"Error: {info.error_message or 'Unknown error'}",
                    f"Attempts: {info.startup_attempts}",
                ],
                color="error",
            )
        else:
            return StatusInfo(
                status="⚪ Idle",
                details=["Ready to start"],
                color="info",
            )
