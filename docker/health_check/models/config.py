"""Service configuration data models."""

from dataclasses import dataclass


@dataclass
class ServiceConfig:
    """Configuration for a monitored service."""

    name: str
    container_name: str
    health_endpoint: str
    port: int
    timeout: int = 10
    retries: int = 3
    dependencies: list[str] | None = None
    critical: bool = True
    networks: list[str] | None = None
    service_discovery: dict[str, str] | None = None

    def __post_init__(self) -> None:
        """Initialize default values after dataclass creation."""
        if self.dependencies is None:
            self.dependencies = []
        if self.networks is None:
            self.networks = []
        if self.service_discovery is None:
            self.service_discovery = {}
