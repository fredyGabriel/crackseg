"""Network status card for TensorBoard component."""

import time
from typing import Any

from .base_card import BaseStatusCard, StatusInfo


class NetworkStatusCard(BaseStatusCard):
    """Status card for TensorBoard network connectivity."""

    def __init__(self) -> None:
        """Initialize network status card."""
        super().__init__("🌐 Network")

    def collect_status_info(
        self, manager: Any, session_manager: Any
    ) -> StatusInfo:
        """Collect network connectivity status information.

        Args:
            manager: TensorBoard manager instance.
            session_manager: Session state manager.

        Returns:
            StatusInfo with network details.
        """
        if not manager.is_running:
            return StatusInfo(
                status="⚪ Not Available",
                details=["Process not running"],
                color="info",
            )

        info = manager.info
        if not info.url:
            return StatusInfo(
                status="🔴 No URL",
                details=["URL not available"],
                color="error",
            )

        # Check network connectivity (simplified version)
        try:
            import urllib.request

            # Quick connectivity test
            start_time = time.time()
            urllib.request.urlopen(info.url, timeout=5)
            latency = (time.time() - start_time) * 1000  # Convert to ms

            if latency < 100:
                latency_status = "🟢 Excellent"
                color = "success"
            elif latency < 500:
                latency_status = "🟡 Good"
                color = "warning"
            else:
                latency_status = "🟠 Slow"
                color = "warning"

            return StatusInfo(
                status="🟢 Connected",
                details=[
                    f"URL: {info.url}",
                    f"Latency: {latency:.0f}ms",
                    latency_status,
                ],
                color=color,
            )

        except Exception:
            return StatusInfo(
                status="🔴 Connection Failed",
                details=[
                    f"URL: {info.url}",
                    "Cannot reach interface",
                ],
                color="error",
            )
