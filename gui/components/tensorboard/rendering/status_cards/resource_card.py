"""Resource status card for TensorBoard component."""

from pathlib import Path
from typing import Any

from .base_card import BaseStatusCard, StatusInfo


class ResourceStatusCard(BaseStatusCard):
    """Status card for TensorBoard resource usage."""

    def __init__(self) -> None:
        """Initialize resource status card."""
        super().__init__("📊 Resources")

    def collect_status_info(
        self, manager: Any, session_manager: Any
    ) -> StatusInfo:
        """Collect resource usage status information.

        Args:
            manager: TensorBoard manager instance.
            session_manager: Session state manager.

        Returns:
            StatusInfo with resource details.
        """
        if not manager.is_running:
            return StatusInfo(
                status="⚪ Not Available",
                details=["Process not running"],
                color="info",
            )

        details = []

        # Port information
        if manager.info.port:
            details.append(f"Port: {manager.info.port}")

        # Log directory size (if available)
        log_dir = session_manager.get_value("log_directory")
        if log_dir and Path(log_dir).exists():
            try:
                # Calculate log directory size
                total_size = sum(
                    f.stat().st_size
                    for f in Path(log_dir).rglob("*")
                    if f.is_file()
                )
                if total_size > 0:
                    from gui.components.tensorboard.utils.formatters import (  # noqa: E501
                        format_file_size,
                    )

                    details.append(f"Logs: {format_file_size(total_size)}")
            except Exception:
                details.append("Logs: Unknown size")

        # Startup attempts
        attempts = manager.info.startup_attempts
        if attempts > 0:
            details.append(f"Restarts: {attempts}")

        # Determine status based on resource usage
        if attempts > 3:
            status = "🟡 High Restarts"
            color = "warning"
        elif details:
            status = "🟢 Normal"
            color = "success"
        else:
            status = "ℹ️ Monitoring"
            color = "info"

        return StatusInfo(
            status=status,
            details=details or ["No data available"],
            color=color,
        )
