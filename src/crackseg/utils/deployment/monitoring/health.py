"""Health checking for monitoring system."""

import logging
import time
from typing import TYPE_CHECKING

import requests

if TYPE_CHECKING:
    from .config import HealthCheckConfig, MonitoringResult


class HealthChecker:
    """Health checker for deployment monitoring."""

    def __init__(self) -> None:
        """Initialize health checker."""
        self.logger = logging.getLogger(__name__)
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "CrackSeg-HealthChecker/1.0"}
        )

    def check_health(self, config: "HealthCheckConfig") -> "MonitoringResult":
        """Check health of a service endpoint.

        Args:
            config: Health check configuration

        Returns:
            Monitoring result with health status
        """
        start_time = time.time()
        timestamp = time.time()

        try:
            response = self.session.get(
                config.url, timeout=config.timeout, allow_redirects=False
            )
            response_time_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                return MonitoringResult(
                    success=True,
                    timestamp=timestamp,
                    metrics={"response_time_ms": response_time_ms},
                    health_status="healthy",
                )
            else:
                return MonitoringResult(
                    success=False,
                    timestamp=timestamp,
                    metrics={"response_time_ms": response_time_ms},
                    health_status="unhealthy",
                    error_message=f"HTTP {response.status_code}",
                )

        except requests.exceptions.Timeout:
            return MonitoringResult(
                success=False,
                timestamp=timestamp,
                health_status="unhealthy",
                error_message="Request timeout",
            )
        except requests.exceptions.ConnectionError:
            return MonitoringResult(
                success=False,
                timestamp=timestamp,
                health_status="unhealthy",
                error_message="Connection failed",
            )
        except Exception as e:
            return MonitoringResult(
                success=False,
                timestamp=timestamp,
                health_status="unhealthy",
                error_message=str(e),
            )

    def wait_for_healthy(
        self, config: "HealthCheckConfig", max_wait: int = 300
    ) -> bool:
        """Wait for service to become healthy.

        Args:
            config: Health check configuration
            max_wait: Maximum wait time in seconds

        Returns:
            True if service became healthy, False otherwise
        """
        start_time = time.time()
        consecutive_successes = 0

        while (time.time() - start_time) < max_wait:
            result = self.check_health(config)

            if result.success and result.health_status == "healthy":
                consecutive_successes += 1
                if consecutive_successes >= config.success_threshold:
                    self.logger.info(
                        f"Service became healthy after "
                        f"{time.time() - start_time:.1f}s"
                    )
                    return True
            else:
                consecutive_successes = 0

            time.sleep(config.interval)

        self.logger.warning(
            f"Service did not become healthy within {max_wait}s"
        )
        return False
