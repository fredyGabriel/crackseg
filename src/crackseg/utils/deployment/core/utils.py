"""Core deployment helper functions extracted from manager.

These functions encapsulate traffic switching, health checks, and replica
management to keep the deployment manager focused and smaller.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


def health_check_deployment(deployment_url: str) -> bool:
    try:
        # Simulate health check
        time.sleep(5)
        return True
    except Exception as exc:  # noqa: BLE001 - keep resilient
        logger.warning(f"Health check failed: {exc}")
        return False


def switch_traffic(environment: str, target: str) -> None:
    logger.info(f"Switching traffic to {target}")


def decommission_deployment(deployment_id: str) -> None:
    logger.info(f"Decommissioning deployment {deployment_id}")


def monitor_canary_performance(result: object, **_: object) -> bool:
    # Simulate performance monitoring
    time.sleep(10)
    return True


def update_traffic_split(deployment_url: str, percentage: int) -> None:
    logger.info(f"Updating traffic split to {percentage}%")


def get_current_replicas(environment: str) -> int:
    return 3


def remove_old_replica(environment: str, index: int) -> None:
    logger.info(f"Removing old replica {index}")


def remove_current_deployment(environment: str) -> None:
    logger.info(f"Removing current deployment in {environment}")
