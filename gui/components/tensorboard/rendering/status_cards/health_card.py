"""Health status card for TensorBoard component."""

from typing import Any

from .base_card import BaseStatusCard, StatusInfo


class HealthStatusCard(BaseStatusCard):
    """Status card for TensorBoard health monitoring."""

    def __init__(self) -> None:
        """Initialize health status card."""
        super().__init__("❤️ Health")

    def collect_status_info(
        self, manager: Any, session_manager: Any
    ) -> StatusInfo:
        """Collect health monitoring status information.

        Args:
            manager: TensorBoard manager instance.
            session_manager: Session state manager.

        Returns:
            StatusInfo with health details.
        """
        info = manager.info

        if not manager.is_running:
            return StatusInfo(
                status="⚪ Not Available",
                details=["Process not running"],
                color="info",
            )

        # Check health check recency
        health_age = info.get_health_age()
        is_stale = info.is_health_check_stale()

        if info.health_status and not is_stale:
            age_str = f"{health_age:.0f}s ago" if health_age else "Just now"
            return StatusInfo(
                status="🟢 Healthy",
                details=[
                    f"Last check: {age_str}",
                    "All systems operational",
                ],
                color="success",
            )
        elif is_stale:
            age_str = f"{health_age:.0f}s ago" if health_age else "Never"
            return StatusInfo(
                status="🟡 Stale",
                details=[
                    f"Last check: {age_str}",
                    "Health check outdated",
                ],
                color="warning",
            )
        else:
            return StatusInfo(
                status="🔴 Unhealthy",
                details=[
                    "Health check failed",
                    "Service may be unresponsive",
                ],
                color="error",
            )
